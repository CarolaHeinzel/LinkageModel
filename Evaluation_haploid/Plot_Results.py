import numpy as np
import matplotlib.pyplot as plt
# Code to create the power plots in the paper

d_list = [0.1, 1, 2, 5] # different values for d
Ms = [50, 100, 200, 500, 1000, 10000] # values for M
r_loop = [0.1, 1, 10] # values for r

H1 = np.array([
    [0.00, 0.05, 0.03, 0.00],
    [0.04, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00],

    [0.05, 0.13, 0.07, 0.01],
    [0.20, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00],

    [0.33, 0.46, 0.24, 0.09],
    [0.44, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00],

    [0.66, 0.71, 0.72, 0.20],
    [0.76, 0.04, 0.00, 0.00],
    [0.04, 0.00, 0.00, 0.00],

    [0.89, 0.97, 0.87, 0.59],
    [0.96, 0.15, 0.00, 0.00],
    [0.09, 0.00, 0.00, 0.00],

    [1.00, 1.00, 1.00, 1.00],
    [1.00, 0.89, 0.13, 0.00],
    [0.95, 0.00, 0.00, 0.00],
])

H0 = np.array([
    [1.00, 1.00, 1.00, 1.00],
    [1.00, 1.00, 1.00, 1.00],
    [1.00, 1.00, 1.00, 1.00],

    [1.00, 0.99, 0.99, 1.00],
    [1.00, 1.00, 1.00, 1.00],
    [1.00, 1.00, 1.00, 1.00],

    [1.00, 0.98, 0.98, 1.00],
    [0.99, 1.00, 1.00, 1.00],
    [1.00, 1.00, 1.00, 1.00],

    [1.00, 1.00, 0.99, 0.98],
    [1.00, 0.99, 1.00, 1.00],
    [1.00, 1.00, 1.00, 1.00],

    [0.99, 1.00, 1.00, 1.00],
    [1.00, 1.00, 1.00, 1.00],
    [0.99, 1.00, 1.00, 1.00],

    [1.00, 1.00, 1.00, 1.00],
    [1.00, 1.00, 0.99, 1.00],
    [0.99, 1.00, 1.00, 1.00],
])

assert H1.shape == (len(Ms)*len(r_loop), len(d_list))
assert H0.shape == (len(Ms)*len(r_loop), len(d_list))

r_labels = ["∞", "10", "1", "0.1"]
x = np.arange(len(r_labels))

M_labels = []
seen = {}
for m in Ms:
    seen[m] = seen.get(m, 0) + 1
    M_labels.append(f"M={m}" if seen[m] == 1 else f"M={m} ({seen[m]})")

# rejection rates table Y[nM, 4, nD]
Y = np.zeros((len(Ms), 4, len(d_list)))
for i_m in range(len(Ms)):
    sl = slice(i_m*3, i_m*3 + 3)  # r=0.1,1,10
    y_inf = 1.0 - H0[sl].mean(axis=0)         # r=∞
    y_r01, y_r1, y_r10 = H1[sl]               # finite r in loop order
    y_finite = np.vstack([y_r10, y_r1, y_r01])# reorder to [10,1,0.1]
    Y[i_m, 0, :] = y_inf
    Y[i_m, 1:, :] = y_finite

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
})

cmap = plt.cm.get_cmap("GnBu")
colors = [
    "#08306B", 
    "#2171B5",  
    "#41B6C4", 
    "#1C9099",  
    "#2CA25F",  
    "#006D2C", 
]
for j_d, d in enumerate(d_list):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.axhline(0.05, linestyle="--", linewidth=1.2, alpha=0.9)

    for i_m, lab in enumerate(M_labels):
        ax.plot(
            x, Y[i_m, :, j_d],
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=colors[i_m],
            label=lab
        )

    ax.set_xticks(x, r_labels)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("r")
    ax.set_ylabel("Proportion reject H0")
    ax.grid(True, alpha=0.25)
    ax.legend(ncols=2, frameon=False)
    fig.tight_layout()

plt.show()



