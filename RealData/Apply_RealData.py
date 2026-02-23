import sys
import gzip
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

VCF_PATH = r"all_chr.biallelic.every2k.maf5.vcf.gz"
DIST_CSV = r"GeneticDistances_1000G.csv"

AF_FILES = {
    "EUR": r"EUR_AF.txt",
    "AFR": r"AFR_AF.txt",
    "EAS": r"EAS_AF.txt",
    "SAS": r"SAS_AF.txt",
    "AMR": r"AMR_AF.txt",
}

# Population order = rows in p (K populations)
POP_ORDER = ["EUR", "AFR", "EAS", "SAS", "AMR"]
K = len(POP_ORDER)

# Marker / numeric guards
DROP_DUPLICATE_POSITIONS = True
EPS = 1e-4                    # clip allele freqs to [eps, 1-eps]
CHROM_BREAK_CM = 1e6          # large jump at chromosome switches (HMM reset)
MIN_MARKERS_PER_SAMPLE = 100  # skip samples with too much missingness

# Test parameters 
N_EM_ITER = 80
LM_MAXITER = 40

# Output
OUT_RESULTS = "LRT_results_all_samples.csv"
import Test_LM_AM_msprime as test


# ============================================================
# Helpers: load allele frequencies
# ============================================================
def load_af_txt(path: str) -> pd.DataFrame:
    """
    Expected format per line (whitespace separated):
        chrom   pos   freq
    Example:
        10  350693  0.136183
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["Chr", "Position", "AF"])
    df["Chr"] = df["Chr"].astype(int)
    df["Position"] = df["Position"].astype(int)
    df["AF"] = df["AF"].astype(float)
    return df


def build_af_table(af_files: Dict[str, str], pop_order: List[str]) -> pd.DataFrame:
    """
    Merge population allele frequencies on (Chr, Position).
    Uses the INTERSECTION across all populations (inner join).
    """
    df = None
    for pop in pop_order:
        tmp = load_af_txt(af_files[pop]).rename(columns={"AF": pop})
        df = tmp if df is None else df.merge(tmp, on=["Chr", "Position"], how="inner")

    if DROP_DUPLICATE_POSITIONS:
        df = df.drop_duplicates(subset=["Chr", "Position"])

    return df.sort_values(["Chr", "Position"]).reset_index(drop=True)


def make_p_matrix(df_af: pd.DataFrame, pop_order: List[str], eps: float) -> np.ndarray:
    """
    Build p matrix of shape (K, M), K=number of populations, M=markers.
    """
    p = df_af[pop_order].to_numpy(dtype=float).T  # (K, M)
    return np.clip(p, eps, 1.0 - eps)


# ============================================================
# Helpers: build cM coordinate from GeneticDistances CSV
# ============================================================
def build_cm_coordinate_from_dist_csv(dist_csv: str) -> Dict[Tuple[int, int], float]:
    """
    Build a per-(Chr, Position) map -> cumulative cM coordinate (relative).
    Uses rows of a distance file containing consecutive SNP pairs:
        Chromosome, Position1, Position2, GeneticDistance_cM
    We accumulate along the chain: cM(pos2) = cM(pos1) + dist.
    """
    df = pd.read_csv(dist_csv)

    required = {"Chromosome", "Position1", "Position2", "GeneticDistance_cM"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Distance CSV missing columns: {missing}. Found: {list(df.columns)}")

    df = df.copy()
    df["Chromosome"] = df["Chromosome"].astype(int)
    df["Position1"] = pd.to_numeric(df["Position1"], errors="coerce")
    df["Position2"] = pd.to_numeric(df["Position2"], errors="coerce")
    df["GeneticDistance_cM"] = pd.to_numeric(df["GeneticDistance_cM"], errors="coerce")

    cm_map: Dict[Tuple[int, int], float] = {}

    for chrom in sorted(df["Chromosome"].dropna().unique()):
        sub = df[df["Chromosome"] == chrom].copy()
        sub = sub[sub["Position2"].notna() & sub["GeneticDistance_cM"].notna()]
        sub = sub[sub["GeneticDistance_cM"] >= 0]
        sub = sub.sort_values(["Position1", "Position2"])
        if len(sub) == 0:
            continue

        first_pos1 = int(sub.iloc[0]["Position1"])
        cm_map[(chrom, first_pos1)] = 0.0

        for _, row in sub.iterrows():
            p1 = int(row["Position1"])
            p2 = int(row["Position2"])
            dist = float(row["GeneticDistance_cM"])

            if (chrom, p1) not in cm_map:
                # If the chain restarts, reset at p1
                cm_map[(chrom, p1)] = 0.0

            cm_map[(chrom, p2)] = cm_map[(chrom, p1)] + dist

    return cm_map


def build_d_step(markers: pd.DataFrame, cm_map: Dict[Tuple[int, int], float], chrom_break_cm: float) -> np.ndarray:
    """
    markers: DataFrame with columns ["Chr", "Position"] in sorted order.
    Returns d_step[i] = cM distance to previous marker, or chrom_break_cm on chromosome switch.
    d_step[0] = 0.
    """
    M = len(markers)
    d_step = np.zeros(M, dtype=float)
    if M == 0:
        return d_step

    prev_chr = int(markers.loc[0, "Chr"])
    prev_pos = int(markers.loc[0, "Position"])
    d_step[0] = 0.0

    if (prev_chr, prev_pos) not in cm_map:
        raise ValueError(f"First marker (Chr={prev_chr}, Pos={prev_pos}) not found in cm_map")

    prev_cm = cm_map[(prev_chr, prev_pos)]

    for i in range(1, M):
        chr_i = int(markers.loc[i, "Chr"])
        pos_i = int(markers.loc[i, "Position"])

        if (chr_i, pos_i) not in cm_map:
            raise ValueError(f"Marker (Chr={chr_i}, Pos={pos_i}) not found in cm_map (distance mismatch)")

        cm_i = cm_map[(chr_i, pos_i)]
        d_step[i] = float(chrom_break_cm) if chr_i != prev_chr else float(abs(cm_i - prev_cm))

        prev_chr, prev_pos, prev_cm = chr_i, pos_i, cm_i

    return d_step


def condense_d_for_mask(d_step: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """
    If a sample has missing genotypes, we drop markers -> d must be condensed:
      d_sub[j] = sum of original d_step between consecutive kept markers.
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
# VCF reading (NO cyvcf2, NO pysam): stream parse with gzip
# ============================================================
def _open_text_auto(path: str):
    """Open plain text or .gz as a text file handle."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def _parse_chrom_to_int(chrom: str) -> Optional[int]:
    """
    Convert CHROM field to integer if possible.
    Accepts '10' or 'chr10'. Returns None for non-numeric chromosomes (X/Y/MT).
    """
    c = chrom.strip()
    if c.startswith("chr") or c.startswith("CHR"):
        c = c[3:]
    # Only numeric chromosomes supported here
    if c.isdigit():
        return int(c)
    return None


def _gt_to_dosage(gt_str: str) -> int:
    """
    Convert GT (e.g., '0/1', '1|1', './.') to dosage in {0,1,2} or -1 if missing.
    Assumes diploid and biallelic.
    """
    if gt_str is None or gt_str == "." or gt_str == "./." or gt_str == ".|.":
        return -1

    # Split on / or |
    if "/" in gt_str:
        a, b = gt_str.split("/", 1)
    elif "|" in gt_str:
        a, b = gt_str.split("|", 1)
    else:
        # Unexpected format
        return -1

    if a == "." or b == ".":
        return -1
    try:
        ai = int(a)
        bi = int(b)
    except ValueError:
        return -1

    # For biallelic SNPs, ai,bi should be 0 or 1. If not, mark missing.
    if ai < 0 or bi < 0 or ai > 1 or bi > 1:
        return -1

    return ai + bi


def read_genotypes_all_samples_stream(vcf_path: str, markers: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    """
    Stream-parse a VCF/VCF.GZ and extract genotypes for the given marker set.

    Returns:
      samples: list of sample IDs
      G: (n_samples, M) genotype counts in {0,1,2}, or -1 for missing / not found

    Notes:
    - This is a pure-Python parser. It will be slower than cyvcf2/pysam on large VCFs.
    - It scans the file once (no random access).
    """
    # Build mapping (chr,pos) -> marker index, and per-chr position sets for membership checks
    M = len(markers)
    key_to_idx: Dict[Tuple[int, int], int] = {}
    pos_by_chr: Dict[int, set] = {}

    for idx, (chr_val, pos_val) in enumerate(markers[["Chr", "Position"]].itertuples(index=False, name=None)):
        chr_i = int(chr_val)
        pos_i = int(pos_val)
        key_to_idx[(chr_i, pos_i)] = idx
        pos_by_chr.setdefault(chr_i, set()).add(pos_i)

    found = np.zeros(M, dtype=bool)
    found_count = 0

    samples: List[str] = []
    G: Optional[np.ndarray] = None

    with _open_text_auto(vcf_path) as f:
        for line in f:
            if not line:
                continue
            if line.startswith("##"):
                continue

            if line.startswith("#CHROM"):
                header = line.rstrip("\n").split("\t")
                samples = header[9:]
                n = len(samples)
                G = np.full((n, M), -1, dtype=np.int8)
                continue

            # Skip until header was read
            if G is None:
                continue

            parts = line.rstrip("\n").split("\t")
            if len(parts) < 10:
                continue

            chrom_raw = parts[0]
            pos = int(parts[1])
            ref = parts[3]
            alt = parts[4]
            fmt = parts[8]
            sample_fields = parts[9:]

            chrom_int = _parse_chrom_to_int(chrom_raw)
            if chrom_int is None:
                continue

            # Keep only markers we care about
            pos_set = pos_by_chr.get(chrom_int)
            if pos_set is None or pos not in pos_set:
                continue

            # Biallelic SNP filter
            if "," in alt:
                continue
            if len(ref) != 1 or len(alt) != 1:
                continue

            col = key_to_idx.get((chrom_int, pos))
            if col is None:
                continue

            # Avoid double-filling duplicates
            if found[col]:
                continue

            # Find GT index in FORMAT
            fmt_keys = fmt.split(":")
            try:
                gt_idx = fmt_keys.index("GT")
            except ValueError:
                # No GT field
                continue

            # Fill this marker for all samples
            # (If a sample is missing or malformed -> -1)
            for s_idx, s_val in enumerate(sample_fields):
                fields = s_val.split(":")
                gt = fields[gt_idx] if gt_idx < len(fields) else None
                G[s_idx, col] = _gt_to_dosage(gt)

            found[col] = True
            found_count += 1

            # Early exit if all markers were found
            if found_count == M:
                break

    return samples, G if G is not None else ([], np.empty((0, M), dtype=np.int8))


# ============================================================
# Run diploid test for all samples
# ============================================================
def run_test_all_samples(samples: List[str], G: np.ndarray, p: np.ndarray, d_step: np.ndarray) -> pd.DataFrame:
    results = []
    M = p.shape[1]
    if G.shape[1] != M:
        raise ValueError("G and p must have the same number of markers")

    for i, sid in enumerate(samples):
        print("i", i)
        g = G[i, :].astype(int)

        keep = (g >= 0)
        m_keep = int(keep.sum())
        if m_keep < MIN_MARKERS_PER_SAMPLE:
            results.append(
                {
                    "sample": sid,
                    "markers_used": m_keep,
                    "status": "skipped_too_many_missing",
                }
            )
            continue

        X = g[keep]                            # (m_keep,)
        p_sub = p[:, keep]                     # (K, m_keep)
        d_sub = condense_d_for_mask(d_step, keep)  # (m_keep,)

        res = test.test_summary(
            X, p_sub, d_sub, K,
            ploidy=2,
            n_em_iter=N_EM_ITER,
            lm_maxiter=LM_MAXITER
        )
        print(bool(res.get("reject_H0", False)))
        row = {
            "sample": sid,
            "markers_used": m_keep,
            "reject_H0": bool(res.get("reject_H0", False)),
            "status": "ok",
        }
        for k in ["LR", "p_value", "ll_AM", "ll_LM", "q_hat_AM", "q_hat_LM"]:
            if k in res:
                row[k] = res[k]
        results.append(row)

        if (i + 1) % 50 == 0:
            print(f"done {i+1}/{len(samples)}")

    return pd.DataFrame(results)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # 1) Allele frequencies -> marker panel
    df_af = build_af_table(AF_FILES, POP_ORDER)
    print("AF table markers:", len(df_af))

    # 2) Distances -> cM map -> d_step
    cm_map = build_cm_coordinate_from_dist_csv(DIST_CSV)

    # Keep only markers that exist in the distance map
    mask_in_map = [
        (int(r.Chr), int(r.Position)) in cm_map
        for r in df_af.itertuples(index=False)
    ]
    df_af = df_af.loc[mask_in_map].reset_index(drop=True)
    print("Markers after distance-map filter:", len(df_af))

    p = make_p_matrix(df_af, POP_ORDER, EPS)
    d_step = build_d_step(df_af[["Chr", "Position"]], cm_map, CHROM_BREAK_CM)

    # 3) Read VCF genotypes for all samples (pure Python streaming)
    samples, G = read_genotypes_all_samples_stream(VCF_PATH, df_af[["Chr", "Position"]])
    print("VCF samples:", len(samples), "Genotype matrix:", G.shape)

    # Optionally drop markers that were never found in the VCF
    found_any = (G >= 0).any(axis=0)
    if not found_any.all():
        n_drop = int((~found_any).sum())
        print(f"Dropping {n_drop} markers not present in VCF")
        df_af = df_af.loc[found_any].reset_index(drop=True)
        p = p[:, found_any]
        d_step = d_step[found_any]
        G = G[:, found_any]

    # 4) Run the test for all individuals
    df_out = run_test_all_samples(samples, G, p, d_step)
    df_out.to_csv(OUT_RESULTS, index=False)
    print("Saved:", OUT_RESULTS)
