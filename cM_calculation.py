import pandas as pd
# https://github.com/dnapainter/apis
def load_genetic_map(filepath):
    df = pd.read_csv(filepath, sep='\t', compression='gzip')
    return df

def interpolate_cM(genetic_map, position):
    # Passende Spaltennamen verwenden
    position_col = 'Position(bp)'
    cm_col = 'Map(cM)'

    # Sicherheitshalber sortieren
    genetic_map = genetic_map.sort_values(by=position_col)

    # Nächstliegende Positionen finden
    before = genetic_map[genetic_map[position_col] <= position].iloc[-1]
    after = genetic_map[genetic_map[position_col] >= position].iloc[0]

    if before[position_col] == after[position_col]:
        return before[cm_col]

    # Lineare Interpolation
    x0, y0 = before[position_col], before[cm_col]
    x1, y1 = after[position_col], after[cm_col]
    return y0 + ((position - x0) / (x1 - x0)) * (y1 - y0)

def compute_genetic_distance(file_path, pos1, pos2):
    genetic_map = load_genetic_map(file_path)
    cm1 = interpolate_cM(genetic_map, pos1)
    cm2 = interpolate_cM(genetic_map, pos2)
    return abs(cm2 - cm1)

# Example
file_path = "/home/ch1158/Ergebnisse_Paper/HMM/geneticMap-GRCh37-master/genetic_map_GRCh37_chr1.txt.gz"  # Pfad zur Datei
pos1 = 101709563
pos2 = 151122489
distance_cm = compute_genetic_distance(file_path, pos1, pos2)
print(f"Genetic distance: {distance_cm:.2f} cM")
#%%
# Data from the paper
snps = [
    {'rsID': 'rs3737576', 'Chr': 1, 'Position': 101709563, 'Fst': 0.44},
    {'rsID': 'rs7554936', 'Chr': 1, 'Position': 151122489, 'Fst': 0.39},
    {'rsID': 'rs2814778', 'Chr': 1, 'Position': 159174683, 'Fst': 0.82},
    {'rsID': 'rs798443', 'Chr': 2, 'Position': 7968275, 'Fst': 0.34},
    {'rsID': 'rs1876482', 'Chr': 2, 'Position': 17362568, 'Fst': 0.75},
    {'rsID': 'rs1834619', 'Chr': 2, 'Position': 17901485, 'Fst': 0.50},
    {'rsID': 'rs3827760', 'Chr': 2, 'Position': 109513601, 'Fst': 0.71},
    {'rsID': 'rs260690', 'Chr': 2, 'Position': 109579738, 'Fst': 0.49},
    {'rsID': 'rs6754311', 'Chr': 2, 'Position': 136707982, 'Fst': 0.41},
    {'rsID': 'rs10497191', 'Chr': 2, 'Position': 158667217, 'Fst': 0.54},
    {'rsID': 'rs12498138', 'Chr': 3, 'Position': 121459589, 'Fst': 0.48},
    {'rsID': 'rs4833103', 'Chr': 4, 'Position': 38815502, 'Fst': 0.37},
    {'rsID': 'rs1229984', 'Chr': 4, 'Position': 100239319, 'Fst': 0.43},
    {'rsID': 'rs3811801', 'Chr': 4, 'Position': 100244319, 'Fst': 0.45},
    {'rsID': 'rs7657799', 'Chr': 4, 'Position': 105375423, 'Fst': 0.44},
    {'rsID': 'rs16891982', 'Chr': 5, 'Position': 33951693, 'Fst': 0.69},
    {'rsID': 'rs7722456', 'Chr': 5, 'Position': 170202984, 'Fst': 0.20},
    {'rsID': 'rs870347', 'Chr': 6, 'Position': 6845035, 'Fst': 0.35},
    {'rsID': 'rs3823159', 'Chr': 6, 'Position': 136482727, 'Fst': 0.50},
    {'rsID': 'rs192655', 'Chr': 7, 'Position': 90518278, 'Fst': 0.21},
    {'rsID': 'rs917115', 'Chr': 8, 'Position': 28172586, 'Fst': 0.35},
    {'rsID': 'rs1462906', 'Chr': 8, 'Position': 31896592, 'Fst': 0.54},
    {'rsID': 'rs6990312', 'Chr': 8, 'Position': 110602317, 'Fst': 0.34},
    {'rsID': 'rs2196051', 'Chr': 8, 'Position': 122124302, 'Fst': 0.43},
    {'rsID': 'rs1871534', 'Chr': 8, 'Position': 145639681, 'Fst': 0.48},
    {'rsID': 'rs3814134', 'Chr': 9, 'Position': 127267689, 'Fst': 0.47},
    {'rsID': 'rs4918664', 'Chr': 10, 'Position': 94921065, 'Fst': 0.53},
    {'rsID': 'rs174570', 'Chr': 11, 'Position': 61597212, 'Fst': 0.51},
    {'rsID': 'rs1079597', 'Chr': 11, 'Position': 113296286, 'Fst': 0.16},
    {'rsID': 'rs2238151', 'Chr': 12, 'Position': 112211833, 'Fst': 0.36},
    {'rsID': 'rs671', 'Chr': 12, 'Position': 112241766, 'Fst': 0.22},
    {'rsID': 'rs7997709', 'Chr': 13, 'Position': 34847737, 'Fst': 0.37},
    {'rsID': 'rs1572018', 'Chr': 13, 'Position': 41715282, 'Fst': 0.41},
    {'rsID': 'rs2166624', 'Chr': 13, 'Position': 42579985, 'Fst': 0.30},
    {'rsID': 'rs7326934', 'Chr': 13, 'Position': 49070512, 'Fst': 0.54},
    {'rsID': 'rs9522149', 'Chr': 13, 'Position': 111827167, 'Fst': 0.44},
    {'rsID': 'rs200354', 'Chr': 14, 'Position': 99375321, 'Fst': 0.32},
    {'rsID': 'rs1800414', 'Chr': 15, 'Position': 28197037, 'Fst': 0.57},
    {'rsID': 'rs12913832', 'Chr': 15, 'Position': 28365618, 'Fst': 0.52},
    {'rsID': 'rs12439433', 'Chr': 15, 'Position': 36220035, 'Fst': 0.39},
    {'rsID': 'rs735480', 'Chr': 15, 'Position': 45152371, 'Fst': 0.39},
    {'rsID': 'rs1426654', 'Chr': 15, 'Position': 48426484, 'Fst': 0.73},
    {'rsID': 'rs459920', 'Chr': 16, 'Position': 89730827, 'Fst': 0.24},
    {'rsID': 'rs4411548', 'Chr': 17, 'Position': 40658533, 'Fst': 0.14},
    {'rsID': 'rs2593595', 'Chr': 17, 'Position': 41056245, 'Fst': 0.47},
    {'rsID': 'rs17642714', 'Chr': 17, 'Position': 48726132, 'Fst': 0.18},
    {'rsID': 'rs4471745', 'Chr': 17, 'Position': 53568884, 'Fst': 0.27},
    {'rsID': 'rs11652805', 'Chr': 17, 'Position': 62987151, 'Fst': 0.39},
    {'rsID': 'rs2042762', 'Chr': 18, 'Position': 35277622, 'Fst': 0.43},
    {'rsID': 'rs7226659', 'Chr': 18, 'Position': 40488279, 'Fst': 0.40},
    {'rsID': 'rs3916235', 'Chr': 18, 'Position': 67578931, 'Fst': 0.63},
    {'rsID': 'rs4891825', 'Chr': 18, 'Position': 67867663, 'Fst': 0.53},
    {'rsID': 'rs7251928', 'Chr': 19, 'Position': 4077096, 'Fst': 0.47},
    {'rsID': 'rs310644', 'Chr': 20, 'Position': 62159504, 'Fst': 0.58},
    {'rsID': 'rs2024566', 'Chr': 22, 'Position': 41697338, 'Fst': 0.31},
]
df_snps = pd.DataFrame(snps)
#%%

def create_df():
    results = []
    # f"/home/ch1158/Ergebnisse_Paper/HMM/geneticMap-GRCh37-master/genetic_map_GRCh37_chr{i}.txt.gz"
    for i in range(1, 23):  # Chromosomes 1 to 22
        file_path = f"/home/ch1158/Ergebnisse_Paper/HMM/geneticMap-GRCh37-master/genetic_map_GRCh37_chr{i}.txt.gz"
    
        # Filter SNPs for chromosome i and sort by position
        snps_chr = df_snps[df_snps['Chr'] == i].sort_values('Position').reset_index(drop=True)
        
        num_snps = len(snps_chr)
    
        if num_snps <= 1:
            # Only one or zero SNPs present – add with GeneticDistance -1
            if num_snps == 1:
                rs1 = snps_chr.loc[0, 'rsID']
                results.append({
                    'Chromosome': i,
                    'rsID1': rs1,
                    'rsID2': None,
                    'GeneticDistance_cM': -1
                })
            continue  # skip further processing for this chromosome
    
        max_j = min(55, num_snps - 1)  # maximum of 55 pairs or less
        for j in range(max_j):
            pos1 = snps_chr.loc[j, 'Position']
            pos2 = snps_chr.loc[j+1, 'Position']
            rs1 = snps_chr.loc[j, 'rsID']
            rs2 = snps_chr.loc[j+1, 'rsID']
            dist_cM = compute_genetic_distance(file_path, pos1, pos2)
            results.append({
                'Chromosome': i,
                'rsID1': rs1,
                'rsID2': rs2,
                'GeneticDistance_cM': dist_cM
            })
    return pd.DataFrame(results)


df_results_try = create_df()
            
