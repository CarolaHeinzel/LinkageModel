import numpy as np
import pandas as pd


import sys

import Test_LM_AM_msprime as test
import Simulation_Linkage_msprime as sL



VCF_PATH = r"all_chr.biallelic.every2k.maf5.vcf.gz"
DIST_CSV = r"GeneticDistances_1000G.csv"

AF_FILES = {
    "EUR": r"EUR_AF.txt",
    "AFR": r"AFR_AF.txt",
    "EAS": r"EAS_AF.txt",
    "SAS": r"SAS_AF.txt",
    "AMR": r"AMR_AF.txt",
}


# Order of populations in q / p
POP_ORDER = ["EUR", "AFR", "EAS", "SAS", "AMR"]
K = len(POP_ORDER)

# Numerical / model settings
EPS = 1e-4
CHROM_BREAK_CM = 1e6          # big jump at chromosome boundary (HMM "reset")
RECOMB_RATE_FALLBACK = 1e-8   # fallback if distance pair missing: 100*r*bp -> cM

# Bootstrap / optimization settings
B = 200
SEED = 0
EM_MAX_ITER = 500
N_EM_ITER = 80
LM_MAXITER = 40

# Which individual to analyze (0 = first sample in VCF)
SAMPLE_INDEX = 0

# ============================================================
# Helpers: AF loading and merge
# ============================================================
def load_af_txt(path: str) -> pd.DataFrame:
    """
    Expected format per line: chrom  position  allele_frequency
    e.g. 10  350693  0.136183   (whitespace or tabs)
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["Chr", "Position", "AF"])
    df["Chr"] = df["Chr"].astype(int)
    df["Position"] = df["Position"].astype(int)
    df["AF"] = df["AF"].astype(float)
    return df

def build_af_table(af_files: dict, pop_order: list) -> pd.DataFrame:
    """Inner-join all AF tables on (Chr, Position)."""
    df = None
    for pop in pop_order:
        tmp = load_af_txt(af_files[pop]).rename(columns={"AF": pop})
        df = tmp if df is None else df.merge(tmp, on=["Chr", "Position"], how="inner")
    df = df.drop_duplicates(subset=["Chr", "Position"])
    return df.sort_values(["Chr", "Position"]).reset_index(drop=True)

def make_p_matrix(df_af: pd.DataFrame, pop_order: list, eps: float) -> np.ndarray:
    p = df_af[pop_order].to_numpy(dtype=float).T  # (K, M)
    return np.clip(p, eps, 1.0 - eps)


# ============================================================
# Helpers: Distance loading -> d_step
# ============================================================
def load_pair_distances(dist_csv: str) -> dict:
    """
    Reads GeneticDistances_1000G.csv and builds a lookup:
      dist[(chr, pos1, pos2)] = cM distance
    Supports columns:
      Chromosome, Position1, Position2, GeneticDistance_cM
    """
    df = pd.read_csv(dist_csv)

    required = {"Chromosome", "Position1", "Position2", "GeneticDistance_cM"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Distance CSV missing columns {missing}. Found: {list(df.columns)}")

    df = df.copy()
    df["Chromosome"] = df["Chromosome"].astype(int)
    df["Position1"] = pd.to_numeric(df["Position1"], errors="coerce")
    df["Position2"] = pd.to_numeric(df["Position2"], errors="coerce")
    df["GeneticDistance_cM"] = pd.to_numeric(df["GeneticDistance_cM"], errors="coerce")

    df = df[df["Position2"].notna() & df["GeneticDistance_cM"].notna()].copy()
    df = df[df["GeneticDistance_cM"] >= 0].copy()

    dist = {}
    for r in df.itertuples(index=False):
        ch = int(r.Chromosome)
        p1 = int(r.Position1)
        p2 = int(r.Position2)
        d = float(r.GeneticDistance_cM)
        dist[(ch, p1, p2)] = d
        dist[(ch, p2, p1)] = d  # allow both directions
    return dist

def build_d_step_from_pairs(markers: pd.DataFrame, pair_dist: dict,
                            chrom_break_cm: float, recomb_rate_fallback: float) -> np.ndarray:
    """
    markers must have columns ['Chr','Position'] sorted by Chr,Position.
    d_step[0]=0.
    For same chromosome: use pair distance if present else fallback to constant-rate approx.
    For chrom change: set chrom_break_cm.
    """
    M = len(markers)
    d_step = np.zeros(M, dtype=float)
    if M == 0:
        return d_step

    prev_chr = int(markers.loc[0, "Chr"])
    prev_pos = int(markers.loc[0, "Position"])
    d_step[0] = 0.0

    warned = 0
    for i in range(1, M):
        ch = int(markers.loc[i, "Chr"])
        ps = int(markers.loc[i, "Position"])

        if ch != prev_chr:
            d_step[i] = float(chrom_break_cm)
        else:
            d = pair_dist.get((ch, prev_pos, ps))
            if d is None:
                # fallback: constant recombination rate approximation
                d = 100.0 * recomb_rate_fallback * abs(ps - prev_pos)
                if warned < 10:
                    print(f"[WARN] Missing distance for chr{ch} {prev_pos}->{ps}. Using fallback {d:.6g} cM.")
                    warned += 1
            d_step[i] = float(d)

        prev_chr, prev_pos = ch, ps

    return d_step

def condense_d_for_mask(d_step: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """
    If some markers are missing for the sample, we drop them and "condense" distances:
      d_sub[j] = cumulative_distance[idx[j]] - cumulative_distance[idx[j-1]]
    """
    idx = np.where(keep)[0]
    if idx.size == 0:
        return np.array([], dtype=float)
    c = np.cumsum(d_step)
    d_sub = np.zeros(idx.size, dtype=float)
    d_sub[0] = 0.0
    if idx.size > 1:
        d_sub[1:] = c[idx[1:]] - c[idx[:-1]]
    return d_sub


# ============================================================
# VCF(.gz) parsing without pysam/cyvcf2 (one sample)
# ============================================================
import gzip
import numpy as np
import pandas as pd

def _norm_chr(ch):
    ch = str(ch).strip()
    if ch.lower().startswith("chr"):
        ch = ch[3:]
    try:
        return int(ch)
    except ValueError:
        return None

def read_one_sample_genotypes_vcfgz(
    vcf_gz_path: str,
    markers: pd.DataFrame,              # columns: Chr, Position
    sample_index: int = 0,
    require_biallelic_snp: bool = True,
) -> tuple[str, np.ndarray]:

    markers = markers[["Chr", "Position"]].copy()
    markers["Chr"] = markers["Chr"].astype(int)
    markers["Position"] = markers["Position"].astype(int)

    # ✅ correct: enumerate over rows
    key_to_idx = {}
    for i, (ch, pos) in enumerate(markers.itertuples(index=False, name=None)):
        key_to_idx[(int(ch), int(pos))] = i

    M = len(markers)
    g = np.full(M, -1, dtype=np.int16)

    sample_id = None
    sample_col = None
    gt_field_index = None

    with gzip.open(vcf_gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line:
                continue
            if line.startswith("##"):
                continue

            if line.startswith("#CHROM"):
                header = line.rstrip("\n").split("\t")
                samples = header[9:]
                if len(samples) == 0:
                    raise ValueError("VCF has no sample columns.")
                if sample_index < 0 or sample_index >= len(samples):
                    raise IndexError(f"sample_index={sample_index} out of range (VCF has {len(samples)} samples).")
                sample_id = samples[sample_index]
                sample_col = 9 + sample_index
                continue

            parts = line.rstrip("\n").split("\t")
            if len(parts) < 10:
                continue

            ch = _norm_chr(parts[0])
            if ch is None:
                continue
            pos = int(parts[1])

            idx = key_to_idx.get((ch, pos))
            if idx is None:
                continue  # not a marker we care about

            ref = parts[3]
            alt = parts[4]

            if require_biallelic_snp:
                if "," in alt:
                    continue
                if len(ref) != 1 or len(alt) != 1:
                    continue

            fmt = parts[8].split(":")
            if gt_field_index is None:
                try:
                    gt_field_index = fmt.index("GT")
                except ValueError:
                    raise ValueError("VCF FORMAT has no GT field.")

            sample_str = parts[sample_col]
            fields = sample_str.split(":")
            if gt_field_index >= len(fields):
                g[idx] = -1
                continue

            gt = fields[gt_field_index]
            if gt in (".", "./.", ".|."):
                g[idx] = -1
                continue

            gt = gt.replace("|", "/")
            alleles = gt.split("/")
            if len(alleles) != 2 or "." in alleles:
                g[idx] = -1
                continue

            try:
                a0 = int(alleles[0])
                a1 = int(alleles[1])
            except ValueError:
                g[idx] = -1
                continue

            # if multi-allelic codes appear (2,3,...) mark missing
            if a0 > 1 or a1 > 1 or a0 < 0 or a1 < 0:
                g[idx] = -1
            else:
                g[idx] = a0 + a1  # 0,1,2

    if sample_id is None:
        raise ValueError("VCF header '#CHROM' not found.")
    return sample_id, g

# ============================================================
# Fit + bootstrap (AM + LM), diploid
# ============================================================
def _simplex_fix(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q = np.asarray(q, float)
    q = np.clip(q, eps, np.inf)
    return q / q.sum()

def fit_am(X, p_sub, K, em_max_iter=500):
    if hasattr(test, "mle_q_admixture_EM_diploid"):
        q = test.mle_q_admixture_EM_diploid(X, p_sub, K, max_iter=em_max_iter)
        return _simplex_fix(q)
    # fallback: use test_summary if needed
    res = test.test_summary(X, p_sub, np.zeros(len(X)), K, ploidy=2, n_em_iter=em_max_iter, lm_maxiter=1)
    return _simplex_fix(res["q_hat_AM"])

def fit_lm(X, p_sub, d_sub, K, n_em_iter=80, lm_maxiter=40):
    res = test.test_summary(X, p_sub, d_sub, K, ploidy=2, n_em_iter=n_em_iter, lm_maxiter=lm_maxiter)
    if "q_hat_LM" not in res or "r_hat" not in res:
        raise KeyError("test_summary must return keys 'q_hat_LM' and 'r_hat' for bootstrap.")
    return _simplex_fix(res["q_hat_LM"]), float(res["r_hat"])

def summarize_q(Q, alpha=0.05):
    se = np.std(Q, axis=0, ddof=1)
    lo = np.quantile(Q, alpha/2, axis=0)
    hi = np.quantile(Q, 1-alpha/2, axis=0)
    return se, lo, hi

def summarize_r(R, alpha=0.05):
    se = float(np.std(R, ddof=1))
    lo = float(np.quantile(R, alpha/2))
    hi = float(np.quantile(R, 1-alpha/2))
    return se, lo, hi

def bootstrap_one_individual(
    X, p_sub, d_sub, pop_order, B=200, seed=0,
    em_max_iter=500, n_em_iter=80, lm_maxiter=40,
    return_replicates=False,
):
    rng = np.random.default_rng(seed)
    K = len(pop_order)
    M = p_sub.shape[1]

    # Fit on observed data
    q_am = fit_am(X, p_sub, K, em_max_iter=em_max_iter)
    q_lm, r_hat = fit_lm(X, p_sub, d_sub, K, n_em_iter=n_em_iter, lm_maxiter=lm_maxiter)

    # AM parametric bootstrap
    theta = np.clip(q_am @ p_sub, 1e-12, 1 - 1e-12)
    Q_am = np.zeros((B, K), float)
    for b in range(B):
        Xb = rng.binomial(2, theta)
        Q_am[b] = fit_am(Xb, p_sub, K, em_max_iter=em_max_iter)

    # LM parametric bootstrap
    Q_lm = np.zeros((B, K), float)
    R_lm = np.zeros(B, float)
    for b in range(B):
        Xb, _ = sL.simulate_markov_chain_diploid(K, q_lm, r_hat, M, d_sub, p_sub, rng=rng)
        qb, rb = fit_lm(Xb, p_sub, d_sub, K, n_em_iter=n_em_iter, lm_maxiter=lm_maxiter)
        Q_lm[b] = qb
        R_lm[b] = rb

    se_am, lo_am, hi_am = summarize_q(Q_am)
    se_lm, lo_lm, hi_lm = summarize_q(Q_lm)
    se_r, lo_r, hi_r = summarize_r(R_lm)

    out = {
        "r_hat_LM": r_hat,
        "r_se_LM": se_r,
        "r_ci_lo_LM": lo_r,
        "r_ci_hi_LM": hi_r,
        "markers_used": int(len(X)),
    }
    for k, pop in enumerate(pop_order):
        out[f"q_hat_AM_{pop}"] = q_am[k]
        out[f"q_se_AM_{pop}"] = se_am[k]
        out[f"q_ci_lo_AM_{pop}"] = lo_am[k]
        out[f"q_ci_hi_AM_{pop}"] = hi_am[k]

        out[f"q_hat_LM_{pop}"] = q_lm[k]
        out[f"q_se_LM_{pop}"] = se_lm[k]
        out[f"q_ci_lo_LM_{pop}"] = lo_lm[k]
        out[f"q_ci_hi_LM_{pop}"] = hi_lm[k]

    if return_replicates:
        return out, {"Q_am": Q_am, "Q_lm": Q_lm, "R_lm": R_lm}

    return out

# ============================================================
# MAIN PIPELINE (sample 0)
# ============================================================
if __name__ == "__main__":
    # 1) Load AFs -> marker panel
    df_af = build_af_table(AF_FILES, POP_ORDER)
    print("Markers from AF intersection:", len(df_af))

    # 2) Load distances and build d_step
    pair_dist = load_pair_distances(DIST_CSV)

    df_af = df_af.sort_values(["Chr", "Position"]).reset_index(drop=True)

    p = make_p_matrix(df_af, POP_ORDER, EPS)
    d_step = build_d_step_from_pairs(
        df_af[["Chr", "Position"]],
        pair_dist,
        chrom_break_cm=CHROM_BREAK_CM,
        recomb_rate_fallback=RECOMB_RATE_FALLBACK,
    )

    print("p shape:", p.shape, "d_step shape:", d_step.shape)
    print("Example d_step[:10]:", d_step[:10])

    # 3) Read genotypes for the first sample from VCF.gz (no pysam/cyvcf2)
    sid, g = read_one_sample_genotypes_vcfgz(
        VCF_PATH,
        df_af[["Chr", "Position"]],
        sample_index=SAMPLE_INDEX,
        require_biallelic_snp=True,
    )
    print("Sample:", sid)
    keep = (g >= 0)
    print("Markers present in VCF for this sample:", int(keep.sum()), "/", len(g))

    # 4) Subset to observed markers for this sample
    X = g[keep].astype(int)
    p_sub = p[:, keep]
    d_sub = condense_d_for_mask(d_step, keep)

    if len(X) < 50:
        raise RuntimeError(f"Too few markers for sample {sid}: {len(X)}")

    # 5) Bootstrap uncertainty of MLEs under AM and LM
    res_boot = bootstrap_one_individual(
        X, p_sub, d_sub,
        pop_order=POP_ORDER,
        B=B, seed=SEED,
        em_max_iter=EM_MAX_ITER,
        n_em_iter=N_EM_ITER,
        lm_maxiter=LM_MAXITER,
    )

    # Print nicely
    print("\n=== Bootstrap results ===")
    print("markers_used:", res_boot["markers_used"])
    print(f"r_hat_LM={res_boot['r_hat_LM']:.6g}  (SE={res_boot['r_se_LM']:.6g}, 95% CI [{res_boot['r_ci_lo_LM']:.6g}, {res_boot['r_ci_hi_LM']:.6g}])")

    for pop in POP_ORDER:
        print(
            f"{pop}: "
            f"q_AM={res_boot[f'q_hat_AM_{pop}']:.4f} "
            f"[{res_boot[f'q_ci_lo_AM_{pop}']:.4f},{res_boot[f'q_ci_hi_AM_{pop}']:.4f}]  |  "
            f"q_LM={res_boot[f'q_hat_LM_{pop}']:.4f} "
            f"[{res_boot[f'q_ci_lo_LM_{pop}']:.4f},{res_boot[f'q_ci_hi_LM_{pop}']:.4f}]"
        )

    # Optionally save
    out_df = pd.DataFrame([{"sample": sid, **res_boot}])
    out_df.to_csv("bootstrap_one_sample.csv", index=False)
    print("\nSaved: bootstrap_one_sample.csv")
#%%
res_boot, boots = bootstrap_one_individual(
    X, p_sub, d_sub, pop_order=POP_ORDER,
    B=200, seed=0,
    em_max_iter=EM_MAX_ITER, n_em_iter=N_EM_ITER, lm_maxiter=LM_MAXITER,
    return_replicates=True,
)
#%%
Q_am = boots["Q_am"]
Q_lm = boots["Q_lm"]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Q_am = boots["Q_am"]   # (B,K)
# Q_lm = boots["Q_lm"]   # (B,K)
R_lm = boots["R_lm"]   # (B,)
POP_ORDER = ["EUR","AFR","EAS","SAS","AMR"]

def plot_cov_heatmap(cov, labels, title, fmt="{:.2e}", figsize=(6.5, 5.5)):
    cov = np.asarray(cov, float)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cov, aspect="auto")  # automatic color scaling

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=13)
    ax.set_yticklabels(labels, fontsize=13)

    # annotate
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            ax.text(j, i, fmt.format(cov[i, j]), ha="center", va="center", fontsize=9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Covariance", fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    ax.set_title(title, fontsize=15)
    ax.tick_params(axis="both", which="major", labelsize=12)
    plt.tight_layout()
    plt.show()


# -----------------------
# 1) Cov(q) for AM and LM
# -----------------------
cov_q_am = np.cov(Q_am, rowvar=False, ddof=1)
cov_q_lm = np.cov(Q_lm, rowvar=False, ddof=1)

plot_cov_heatmap(cov_q_am, POP_ORDER, "Bootstrap covariance of q (Admixture model)")
plot_cov_heatmap(cov_q_lm, POP_ORDER, "Bootstrap covariance of q (Linkage model)")

# -----------------------
# 2) Cov([q, r]) for LM
# -----------------------
R_lm = boots["R_lm"]
Z = np.column_stack([Q_lm, R_lm])          # (B, K+1)
cov_qr_lm = np.cov(Z, rowvar=False, ddof=1)
labels_qr = POP_ORDER + ["r"]

plot_cov_heatmap(cov_qr_lm, labels_qr, "Bootstrap covariance of [q, r] (Linkage model)", figsize=(7.0, 6.0))
