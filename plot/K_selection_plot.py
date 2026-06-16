import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import os
import re

# ================= Configuration =================
result_dir = "./results/data/ablation"
file_pattern = "*_K_ablation_THFBN.pkl"
files = glob.glob(os.path.join(result_dir, file_pattern))
print(f"Found {len(files)} result files")

# ================= Parse band from filename =================
band_pattern = re.compile(r'_(Gamma|HFO)_')

def extract_band(filepath):
    m = band_pattern.search(os.path.basename(filepath))
    return m.group(1) if m else "Unknown"

# Group by band
band_data = {}
for f in files:
    band = extract_band(f)
    if band == "Unknown":
        continue
    with open(f, 'rb') as fp:
        data = pickle.load(fp)
    if band not in band_data:
        band_data[band] = {}
    for K_str, metrics in data.items():
        K = int(K_str)
        if 'stability_mean' not in metrics or 'hub_consistency' not in metrics:
            continue
        if K not in band_data[band]:
            band_data[band][K] = []
        band_data[band][K].append((metrics['stability_mean'], metrics['hub_consistency']))

# ================= Process each band and plot =================
for band in sorted(band_data.keys()):
    all_data = band_data[band]
    ks = sorted(all_data.keys())
    print(f"\nBand: {band}, K list: {ks}")

    # Compute average per K, filter NaN
    stab_means, hub_means = [], []
    for K in ks:
        vals = all_data[K]
        valid = [(s, h) for s, h in vals if not (np.isnan(s) or np.isnan(h))]
        if not valid:
            print(f"  Warning: K={K} has no valid data")
            stab_means.append(np.nan)
            hub_means.append(np.nan)
        else:
            stab_means.append(np.mean([v[0] for v in valid]))
            hub_means.append(np.mean([v[1] for v in valid]))

    stab_means = np.array(stab_means)
    hub_means = np.array(hub_means)
    valid_idx = ~np.isnan(stab_means) & ~np.isnan(hub_means)
    ks_valid = np.array(ks)[valid_idx]
    stab_valid = stab_means[valid_idx]
    hub_valid = hub_means[valid_idx]

    if len(ks_valid) == 0:
        print(f"  Band {band} has no usable data, skipping plot.")
        continue

    # Best K selection
    stab_norm = (stab_valid - stab_valid.min()) / (stab_valid.max() - stab_valid.min() + 1e-8)
    hub_norm = (hub_valid - hub_valid.min()) / (hub_valid.max() - hub_valid.min() + 1e-8)
    score = stab_norm + (1 - hub_norm)
    best_idx = np.argmin(score)
    best_k = ks_valid[best_idx]
    print(f"  Best K = {best_k}, Score = {score[best_idx]:.4f}")

    for k, s, h in zip(ks_valid, stab_valid, hub_valid):
        print(f"    K={k}: Stability={s:.4f}, Consistency={h:.4f}")

    # Plot
    plt.figure(figsize=(5, 5))
    color_map = {2: "#6EAADA", 3: "#DC6133", 4: "#3DB991",
                 5: "#D7191C", 6: "#F2B044", 7: "#B797B5"}
    for i, k in enumerate(ks_valid):
        col = color_map.get(k, "#999999")
        x, y = stab_valid[i], hub_valid[i]
        if k == best_k:
            plt.scatter(x, y, marker='*', s=400, color="#D7191C",
                        edgecolor='black', linewidth=0.7, zorder=3)
        else:
            plt.scatter(x, y, s=120, color=col,
                        edgecolor='black', linewidth=0.7, zorder=2)
        dx = (stab_valid.max() - stab_valid.min()) * 0.08
        dy = (hub_valid.max() - hub_valid.min()) * 0.05
        plt.text(x - dx, y + dy, f'K={k}',
                 fontfamily="Times New Roman", fontsize=16, color='black')

    plt.xlabel("Relative stability", fontfamily="Times New Roman", fontsize=20)
    plt.ylabel("Hub consistency", fontfamily="Times New Roman", fontsize=20)

    if band == 'Gamma':
        x_ticks = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
        y_ticks = np.array([0.16, 0.17, 0.18, 0.19, 0.20])
        x_low, x_high = x_ticks[0] - 0.1, x_ticks[-1] + 0.1
        y_low, y_high = y_ticks[0] - 0.01, y_ticks[-1] + 0.01
    else:
        x_ticks = np.array([1.0, 1.25, 1.5, 1.75])
        y_ticks = np.array([0.24, 0.26, 0.28, 0.30])
        x_low, x_high = x_ticks[0] - 0.25, x_ticks[-1] + 0.25
        y_low, y_high = y_ticks[0] - 0.02, y_ticks[-1] + 0.02

    plt.xlim(x_low, x_high)
    plt.ylim(y_low, y_high)
    plt.xticks(x_ticks, fontfamily="Times New Roman", fontsize=18)
    plt.yticks(y_ticks, fontfamily="Times New Roman", fontsize=18)
    # =================================================

    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    os.makedirs("./results/figures", exist_ok=True)
    save_path = f"./results/figures/Stability_Consistency_{band}.png"
    plt.savefig(save_path, dpi=300)
    print(f"  Figure saved to: {save_path}")
    plt.show()