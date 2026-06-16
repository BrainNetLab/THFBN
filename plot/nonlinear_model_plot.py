import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

# ========== Configuration ==========
bands = ["Gamma", "HFO"]
stages = ["pre", "ictal", "post"]
# Palette consistent with example: Real corresponds to Kernel (nonlinear), Null corresponds to Linear
palette = {"Linear": "#9970AC", "Kernel": "#91C6F2"}
save_dir = "./results/figures"
os.makedirs(save_dir, exist_ok=True)

# ---------- Helper functions ----------
def significance_star(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return "ns"

def nice_axis_limits(all_data, max_ticks=5):
    data_min = np.nanmin(all_data)
    data_max = np.nanmax(all_data)
    if data_max == data_min:
        return data_min - 0.5, data_max + 0.5, np.linspace(data_min, data_max, max_ticks)
    raw_step = (data_max - data_min) / (max_ticks - 1)
    exponent = np.floor(np.log10(raw_step))
    fraction = raw_step / 10 ** exponent
    if fraction <= 1: nice_fraction = 1
    elif fraction <= 2: nice_fraction = 2
    elif fraction <= 5: nice_fraction = 5
    else: nice_fraction = 10
    step = nice_fraction * 10 ** exponent
    ymin = np.floor(data_min / step) * step
    ymax = np.ceil(data_max / step) * step
    if ymin > data_min: ymin -= step
    yticks = np.arange(ymin, ymax + step/2, step)
    return ymin, ymax, yticks

def remove_outliers_iqr(data):
    """Remove outliers beyond 1.5*IQR"""
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    return data[(data >= lower) & (data <= upper)]

# ---------- Plotting function (boxplot version) ----------
def plot_nmse_box(df, band, remove_outliers=True):
    fig, ax = plt.subplots(figsize=(6, 4))
    np.random.seed(42)  # ensure reproducible scatter jitter

    # ----- Build long-format data for boxplot -----
    data_long = []
    for stage in stages:
        sub = df[(df["band"] == band) & (df["stage"] == stage)]
        if sub.empty:
            continue
        lin_vals = sub["nmse_lin_out"].dropna().values
        ker_vals = sub["nmse_kernel_out"].dropna().values

        # Optional outlier filtering (maintain consistency with statistical tests)
        if remove_outliers:
            lin_vals_clean = remove_outliers_iqr(lin_vals)
            ker_vals_clean = remove_outliers_iqr(ker_vals)
        else:
            lin_vals_clean = lin_vals
            ker_vals_clean = ker_vals

        for v in lin_vals_clean:
            data_long.append({"Stage": stage, "Group": "Linear", "NMSE": v})
        for v in ker_vals_clean:
            data_long.append({"Stage": stage, "Group": "Kernel", "NMSE": v})

    df_long = pd.DataFrame(data_long)
    if df_long.empty:
        print(f"No data for band {band}, skip.")
        return

    # ----- Draw boxplot -----
    sns.boxplot(data=df_long, x="Stage", y="NMSE", hue="Group",
                order=stages, hue_order=["Linear", "Kernel"],
                palette=palette, width=0.5, gap=0.25,
                linewidth=1.2, fliersize=0, ax=ax)

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
    ax.legend(handles[:2], ["Linear", "Nonlinear"], loc='upper right', bbox_to_anchor=(1.0, 1.08),
              frameon=False, ncol=2, fontsize=16, handlelength=1.2)

    # ----- Overlay scatter points (jitter) -----
    offset = 0.12  # offset of two box centers from x=i
    jitter_width = 0.055  # jitter width of scatter points within the box
    for i, stage in enumerate(stages):
        sub = df_long[(df_long["Stage"] == stage)]
        lin_vals = sub[sub["Group"] == "Linear"]["NMSE"].values
        ker_vals = sub[sub["Group"] == "Kernel"]["NMSE"].values

        if len(lin_vals) > 0:
            x_lin = i - offset + np.random.uniform(-jitter_width, jitter_width, len(lin_vals))
            ax.scatter(x_lin, lin_vals, s=10, facecolors='white', edgecolors='black',
                       linewidths=0.5, zorder=3)
        if len(ker_vals) > 0:
            x_ker = i + offset + np.random.uniform(-jitter_width, jitter_width, len(ker_vals))
            ax.scatter(x_ker, ker_vals, s=10, facecolors='white', edgecolors='black',
                       linewidths=0.5, zorder=3)

    # ----- Significance annotation (paired t-test) -----
    y_max_data = df_long["NMSE"].max()
    y_min_data = df_long["NMSE"].min()
    y_range = y_max_data - y_min_data if y_max_data != y_min_data else 1.0
    sig_offset = y_range * 0.04
    bar_height = y_range * 0.03

    for i, stage in enumerate(stages):
        sub = df[(df["band"] == band) & (df["stage"] == stage)]
        if sub.empty:
            continue
        lin_raw = sub["nmse_lin_out"].dropna().values
        ker_raw = sub["nmse_kernel_out"].dropna().values

        if remove_outliers:
            lin_vals = remove_outliers_iqr(lin_raw)
            ker_vals = remove_outliers_iqr(ker_raw)
        else:
            lin_vals = lin_raw
            ker_vals = ker_raw

        # Paired t-test (truncate to shorter length, approximate treatment)
        min_len = min(len(lin_vals), len(ker_vals))
        if min_len < 3:
            continue
        _, p = ttest_rel(lin_vals[:min_len], ker_vals[:min_len])
        star = significance_star(p)

        # Annotation line position: between the two boxes of the current stage
        x_lin = i - 0.2
        x_ker = i + 0.2
        base_y = y_max_data + sig_offset
        ax.plot([x_lin, x_lin, x_ker, x_ker],
                [base_y, base_y + bar_height, base_y + bar_height, base_y],
                lw=1.2, c='black')
        ax.text((x_lin + x_ker) / 2, base_y + bar_height * 1.3, star,
                ha='center', va='bottom', fontsize=16, fontweight='bold')

    # ----- Axis styling -----
    ax.set_xlabel("")
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=16)
    ymin, ymax_tick, yticks = nice_axis_limits(df_long["NMSE"].values, max_ticks=5)
    y_top = y_max_data + sig_offset + bar_height * 3
    ax.set_ylim(ymin - 0.05, max(ymax_tick, y_top))
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:g}" for y in yticks], fontsize=16)
    ax.set_ylabel("NMSE", fontsize=18)
    ax.tick_params(direction='out', length=4, width=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout(pad=0.5)

    fname = f"{band}_nmse.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches='tight')
    plt.close()

# ---------- Main program ----------
if __name__ == "__main__":
    df = pd.read_csv("./results/data/nonlinear/local_fit_comparison.csv")
    df = df[(df["band"].isin(bands)) & (df["stage"].isin(stages))]
    print(f"Loaded {len(df)} windows.")

    for band in bands:
        plot_nmse_box(df, band, remove_outliers=True)

    print(f"✅ NMSE boxplots saved to {save_dir}")