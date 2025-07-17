import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Plot the results

lst = [12, 14, 19] # Example List

path = "/home/ch1158/Downloads/1000G_SampleListWithLocations(2).txt"
df = pd.read_csv(path, sep="\t", encoding="utf-8", header = None)

col = df.iloc[:, 2]

# Finde Indizes, bei denen der Wert exakt "EUR" ist
index_list_EUR = col[col == "EUR"].index.tolist()
index_list_EAS = col[col == "EAS"].index.tolist()
index_list_SAS = col[col == "SAS"].index.tolist()
index_list_AFR = col[col == "AFR"].index.tolist()
index_list_AMR = col[col == "AMR"].index.tolist()
        
count = sum(1 for x in lst if x in index_list_EUR)
print(count, len(index_list_EUR))
ount = sum(1 for x in lst if x in index_list_AFR)
print(count, len(index_list_AFR))
count = sum(1 for x in lst if x in index_list_AMR)
print(count, len(index_list_AMR))

count = sum(1 for x in lst if x in index_list_SAS)
print(count, len(index_list_SAS))
count = sum(1 for x in lst if x in index_list_EAS)
print(count, len(index_list_EAS))
#%%

# Results
intervals = [(488, 503),
             (488, 661),
             (300, 347),
             (471, 489),
             (337, 504)]

# Start- und Endwerte extrahieren
starts = [i[0] for i in intervals]
ends = [i[1] for i in intervals]

# Anzahl der Intervalle
n = len(intervals)
x = np.arange(n)  # X-Positionen

bar_width = 0.35  # Breite der Balken

# Plot
plt.figure(figsize=(8, 5))
plt.bar(x - bar_width/2, starts, bar_width, label='Number of Reject H0', color='purple')
plt.bar(x + bar_width/2, ends, bar_width, label='Number of Individuals in total', color='grey')

# Achsen und Labels
plt.xticks(x, ["EUR", "AFR", "AMR", "SAS", "EAS"])
plt.ylabel('Number of Individuals')
plt.legend()
plt.tight_layout()
plt.show()



