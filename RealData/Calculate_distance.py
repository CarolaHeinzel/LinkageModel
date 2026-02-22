import os
import ast
import numpy as np
import pandas as pd

# ---------------------------
# Adjust paths
# ---------------------------
SNP_PATH = r"snp_output.txt"
MAP_DIR  = r"geneticMap-GRCh37-master"
OUT_SNP_CSV = "Distance_1000G_new.csv"
OUT_DIST_CSV = "GeneticDistances_1000G.csv"


# ---------------------------
# Load SNP records
# ---------------------------
def load_snp_records(path: str) -> pd.DataFrame:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(ast.literal_eval(line))  # safer than eval()

    df = pd.DataFrame(records)

    # Expected columns: Chr, Position, rsID (adapt if needed)
    required = {"Chr", "Position", "rsID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"SNP file is missing columns: {missing}. Found: {list(df.columns)}")

    df["Chr"] = df["Chr"].astype(int)
    df["Position"] = df["Position"].astype(int)
    return df


# ---------------------------
# Genetic map loading + robust interpolation
# ---------------------------
def load_genetic_map_chr(map_dir: str, chrom: int) -> pd.DataFrame:
    fp = os.path.join(map_dir, f"genetic_map_GRCh37_chr{chrom}.txt.gz")
    if not os.path.exists(fp):
        raise FileNotFoundError(fp)

    df = pd.read_csv(fp, sep="\t", compression="gzip")

    # Standard column names in the geneticMap-GRCh37 repository
    pos_col = "Position(bp)"
    cm_col = "Map(cM)"
    if pos_col not in df.columns or cm_col not in df.columns:
        raise ValueError(f"Map {fp} has unexpected columns. Found: {list(df.columns)}")

    df = df[[pos_col, cm_col]].dropna()
    df = df.sort_values(pos_col).reset_index(drop=True)
    return df


def interpolate_cM_many(genetic_map: pd.DataFrame, positions_bp: np.ndarray) -> np.ndarray:
    """
    Robust linear interpolation of cM positions for many bp positions.
    Positions outside the map range are clamped to the nearest map boundary
    (instead of raising an IndexError).
    """
    pos_col = "Position(bp)"
    cm_col = "Map(cM)"

    x = genetic_map[pos_col].to_numpy(dtype=float)
    y = genetic_map[cm_col].to_numpy(dtype=float)

    p = np.asarray(positions_bp, dtype=float)

    # searchsorted: index of first x[idx] >= p
    idx = np.searchsorted(x, p, side="left")

    # Clamp indices to [1, len(x)-1] so idx-1 and idx are valid
    idx = np.clip(idx, 1, len(x) - 1)

    x0 = x[idx - 1]
    x1 = x[idx]
    y0 = y[idx - 1]
    y1 = y[idx]

    # Avoid division by zero if x0 == x1
    denom = np.where(x1 == x0, 1.0, (x1 - x0))
    t = (p - x0) / denom
    cm = y0 + t * (y1 - y0)

    # Clamp values outside the map range to boundary cM values
    cm = np.where(p <= x[0], y[0], cm)
    cm = np.where(p >= x[-1], y[-1], cm)

    return cm


# ---------------------------
# Compute distances per chromosome
# ---------------------------
def compute_distances_for_snps(df_snps: pd.DataFrame, map_dir: str) -> pd.DataFrame:
    results = []

    for chrom in range(1, 23):
        # Filter SNPs for this chromosome and sort by position
        snps_chr = df_snps[df_snps["Chr"] == chrom].sort_values("Position").reset_index(drop=True)
        n = len(snps_chr)
        if n == 0:
            continue

        # Load genetic map once per chromosome
        try:
            gmap = load_genetic_map_chr(map_dir, chrom)
        except FileNotFoundError:
            print(f"[WARN] Genetic map for chromosome {chrom} not found — skipping.")
            continue

        # Convert all bp positions to cM using interpolation
        pos = snps_chr["Position"].to_numpy()
        cm = interpolate_cM_many(gmap, pos)

        # Distances between consecutive SNPs: dist[j] = |cm[j+1] - cm[j]|
        dist = np.abs(np.diff(cm))

        # First SNP has no previous neighbor -> mark with -1
        results.append({
            "Chromosome": chrom,
            "rsID1": snps_chr.loc[0, "rsID"],
            "rsID2": None,
            "Position1": int(snps_chr.loc[0, "Position"]),
            "Position2": None,
            "GeneticDistance_cM": -1.0
        })

        # Add distances for consecutive pairs
        for j in range(n - 1):
            results.append({
                "Chromosome": chrom,
                "rsID1": snps_chr.loc[j, "rsID"],
                "rsID2": snps_chr.loc[j + 1, "rsID"],
                "Position1": int(snps_chr.loc[j, "Position"]),
                "Position2": int(snps_chr.loc[j + 1, "Position"]),
                "GeneticDistance_cM": float(dist[j])
            })

    return pd.DataFrame(results)


def distances_per_chromosome(df_dist: pd.DataFrame) -> list[list[float]]:
    """Return a list of lists: one list of distances per chromosome."""
    out = []
    for chrom in sorted(df_dist["Chromosome"].unique()):
        vals = df_dist.loc[df_dist["Chromosome"] == chrom, "GeneticDistance_cM"].tolist()
        out.append(vals)
    return out


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    df_snps = load_snp_records(SNP_PATH)
    df_snps.to_csv(OUT_SNP_CSV, index=False)

    df_dist = compute_distances_for_snps(df_snps, MAP_DIR)
    df_dist.to_csv(OUT_DIST_CSV, index=False)

    d_by_chr = distances_per_chromosome(df_dist)
    print("Chromosomes with distances:", len(d_by_chr))
    print("Example (chr1 first 10):", d_by_chr[0][:10])


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_dist = pd.read_csv("GeneticDistances_1000G.csv")

# Pool distances across all chromosomes, remove invalid/placeholder entries
x = df_dist["GeneticDistance_cM"].to_numpy(dtype=float)
x = x[np.isfinite(x) & (x > 0)]

# Histogram with log-spaced bins (good when values span multiple orders of magnitude)
n_bins = 50
edges = np.logspace(np.log10(x.min()), np.log10(x.max()), n_bins + 1)
counts, _ = np.histogram(x, bins=edges)
centers = np.sqrt(edges[:-1] * edges[1:])   # geometric bin centers
widths = edges[1:] - edges[:-1]

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(centers, counts, width=widths, align="center")
ax.set_xscale("log")

# Larger fonts for axis labels
ax.set_xlabel("Genetic distance (cM)", fontsize=16)
ax.set_ylabel("Number", fontsize=16)

# Larger tick labels on axes
ax.tick_params(axis="both", which="major", labelsize=14)
ax.tick_params(axis="both", which="minor", labelsize=12)

plt.tight_layout()
plt.show()
