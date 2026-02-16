import numpy as np
import pandas as pd


# Calculate the allele frequencies of the true data

VCF_PATH = "1000G_AIMsetKidd.vcf"
PANEL_PATH = "1000G_SampleListWithLocations.txt"
OUT_CSV = "allele_freq_by_superpop.csv"

res_all = []
def read_vcf(path):
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            res_all.append(line)
    return res_all
res_x = read_vcf(VCF_PATH)


res_x_rm = res_x[255:310]

x_all1 = []
x_all2 = []
for i in res_x_rm:
    genotypes = [gt for gt in i.strip().split('\t') if gt]
    left = [item.split('|')[0] for item in genotypes[9:]]
    right = [item.split('|')[1] for item in genotypes[9:]]

    x_all1.append(left) # 55 times 2504 many entries, i.e. [marker, individual]
    x_all2.append(right)

x_all_inv1 = np.array(x_all1).astype(int) #transpose() # phased data [marker, individual]
x_all_inv2 = np.array(x_all2).astype(int) #transpose() # phased  data
x_all_inv1 = x_all_inv1.transpose()
x_all_inv2 = x_all_inv2.transpose()



import numpy as np

SUPERPOPS = ["AFR", "EUR", "EAS", "SAS", "AMR"]

# --- 1) Sample IDs aus dem VCF-Header holen ---
header_line = None
for line in res_x:
    if line.startswith("#CHROM"):
        header_line = line
        break

if header_line is None:
    raise RuntimeError("Keine #CHROM Headerzeile im VCF gefunden.")

header_parts = header_line.strip().split("\t")
vcf_sample_ids = header_parts[9:]   

# --- 2) Panel einlesen und sample -> super_pop Mapping bauen ---
panel = pd.read_csv(PANEL_PATH, sep="\t", header=None, dtype=str)

if panel.shape[1] < 3:
    raise ValueError("Panel-Datei muss mind. 3 Spalten haben: sample, pop, super_pop.")

panel = panel.iloc[:, :4] 
panel.columns = ["sample", "pop", "super_pop", "sex"][:panel.shape[1]]

sample_to_super = dict(zip(panel["sample"], panel["super_pop"]))

super_of_vcf_samples = np.array([sample_to_super.get(s, None) for s in vcf_sample_ids], dtype=object)

matched = sum(sp in SUPERPOPS for sp in super_of_vcf_samples)
print("Matched samples (in SUPERPOPS):", matched, "von", len(vcf_sample_ids))

if matched == 0:
    raise ValueError("Keine VCF Sample IDs matchen die Panel Sample IDs. (IDs stimmen nicht überein)")

meta = []
for line in res_x_rm:
    parts = line.strip().split("\t")
    # VCF: CHROM POS ID REF ALT ...
    meta.append((parts[0], int(parts[1]), parts[2], parts[3], parts[4]))

meta_df = pd.DataFrame(meta, columns=["CHROM", "POS", "ID", "REF", "ALT"])


hap1 = x_all_inv1.astype(float)
hap2 = x_all_inv2.astype(float)

M = hap1.shape[1]
out = meta_df.copy()

for sp in SUPERPOPS:
    idx = np.where(super_of_vcf_samples == sp)[0]
    n = len(idx)

    out[f"N_{sp}"] = n

    if n == 0:
        out[f"AF_{sp}"] = np.nan
        continue

    alt_sum = hap1[idx, :].sum(axis=0) + hap2[idx, :].sum(axis=0)  # (M,)
    out[f"AF_{sp}"] = alt_sum / (2.0 * n)

out.to_csv(OUT_CSV, index=False)
print("Saved allele frequencies to:", OUT_CSV)
print(out.head())



