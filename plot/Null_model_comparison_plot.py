import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from scipy.stats import ttest_rel
from sympy.abc import alpha

from utils import *

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
bands = ["Gamma", "HFO"]
I_list = [16, 20, 24]
sparsity_list = [0.5, 0.6, 0.7, 0.8]
palette = {
    "Real": "#9970AC",
    "Null": "#91C6F2"
}

def plot_rho_band_I(df_sub, band, I):
    n_sample_per_sz = 5  # Number of windows kept per seizure
    np.random.seed(42)   # Ensure reproducibility

    # ----- Downsampling: randomly select n_sample_per_sz windows per seizure -----
    sampled_rows = []
    for (pid, sz), group in df_sub.groupby(['patient', 'seizure']):
        # There may be many windows per seizure, randomly select up to n_sample_per_sz
        n_take = n_sample_per_sz
        take_idx = np.random.choice(group.index, size=n_take, replace=False)
        sampled_rows.append(group.loc[take_idx])
    df_plot = pd.concat(sampled_rows).reset_index(drop=True)

    # ----- Build long-format data for plotting -----
    data_long = []
    for sp in sparsity_list:
        subset = df_plot[df_plot['sparsity'] == sp]
        for val in subset['RHO_real']:
            data_long.append({'Sparsity': sp, 'Group': 'Real', 'RHO': val})
        for val in subset['RHO_null']:
            data_long.append({'Sparsity': sp, 'Group': 'Null', 'RHO': val})
    df_long = pd.DataFrame(data_long)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df_long, x='Sparsity', y='RHO', hue='Group', palette=palette,
                width=0.65, gap=0.25, linewidth=1.2, fliersize=0, ax=ax)

    # Set box borders to black
    for patch in ax.patches:
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
    for line in ax.lines:
        line.set_color('black')
        line.set_linewidth(1.2)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend_.remove()
    ax.legend(handles[:2], labels[:2], frameon=False, loc="upper right",
              ncol=2, fontsize=16, handlelength=1.2)

    # Significance annotation
    max_real = df_long[df_long['Group'] == 'Real']['RHO'].max()
    y_min_data = df_long['RHO'].min()
    y_range = df_long['RHO'].max() - y_min_data
    sig_offset = y_range * 0.18
    bar_height = y_range * 0.02
    base_y = max_real + sig_offset

    np.random.seed(42)
    box_width_fraction = 0.6
    box_patches = ax.patches[:len(sparsity_list) * 2]
    for k, patch in enumerate(box_patches):
        x = patch.get_path().vertices[:, 0]
        x_left = np.min(x)
        x_right = np.max(x)
        x_center = (x_left + x_right) / 2
        x_half = (x_right - x_left) / 2
        jitter_width = x_half * box_width_fraction
        sp_idx = k // 2
        group_idx = k % 2
        sub = df_sub[df_sub["sparsity"] == sparsity_list[sp_idx]]
        if group_idx == 0:
            vals = sub["RHO_real"].values
        else:
            vals = sub["RHO_null"].values
        x_jitter = x_center + np.random.uniform(-jitter_width, jitter_width, len(vals))
        ax.scatter(x_jitter, vals, s=12, facecolors='white', edgecolors='black',
                   linewidths=0.5, zorder=3)

    for i, sp in enumerate(sparsity_list):
        sub = df_sub[df_sub["sparsity"] == sp]
        if len(sub) == 0:
            continue
        _, p = ttest_rel(sub['RHO_real'].values, sub['RHO_null'].values, alternative='greater')
        star = significance_star(p)
        x1 = i - 0.2
        x2 = i + 0.2
        ax.plot([x1, x1, x2, x2], [base_y, base_y + bar_height, base_y + bar_height, base_y],
                lw=1.2, c='black')
        ax.text((x1 + x2) / 2, base_y + bar_height * 1.2, star, ha='center', va='bottom',
                fontsize=16, fontweight='bold')

    # Axes styling
    ax.set_xlabel("")
    ax.set_xticks(range(len(sparsity_list)))
    ax.set_xticklabels([str(x) for x in sparsity_list], fontsize=16)

    ymin, ymax_tick, yticks = nice_axis_limits(df_long["RHO"].values, max_ticks=5)
    y_top = base_y + bar_height * 2
    ax.set_ylim(ymin - 0.05, max(ymax_tick, y_top))
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:g}" for y in yticks], fontsize=16)
    ax.set_ylabel("RHO", fontsize=18)

    ax.tick_params(direction='out', length=4, width=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    # Filename includes I and band
    fname = f"I{I}_{band}.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=600)
    plt.close()

if __name__ == "__main__":
    df = pd.read_csv("./results/data/null_model/RHO_Null_Model_All.csv")
    save_dir = "./results/figures"
    os.makedirs(save_dir, exist_ok=True)

    for I in I_list:
        for band in bands:
            df_sub = df[(df['I'] == I) & (df['band'] == band)].copy()
            if len(df_sub) == 0:
                print(f"No data for I={I}, {band}, skip.")
                continue
            plot_rho_band_I(df_sub, band, I)

    print("✅ All figures saved.")