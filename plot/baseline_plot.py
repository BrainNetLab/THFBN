import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from matplotlib.transforms import Affine2D
from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox

plt.rcParams["font.family"] = "Times New Roman"

# ================= Helper functions =================
def significance_star(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"

def nice_axis_limits(all_data, max_ticks=5):
    data_min = np.nanmin(all_data)
    data_max = np.nanmax(all_data)
    if data_max == data_min:
        return data_min - 0.5, data_max + 0.5, np.linspace(data_min, data_max, max_ticks)

    raw_step = (data_max - data_min) / (max_ticks - 1)
    exponent = np.floor(np.log10(raw_step))
    fraction = raw_step / 10 ** exponent

    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10

    step = nice_fraction * 10 ** exponent
    ymin = np.floor(data_min / step) * step
    ymax = np.ceil(data_max / step) * step
    if ymin > data_min:
        ymin -= step
    yticks = np.arange(ymin, ymax + step / 2, step)
    return ymin, ymax, yticks

def cohens_d(x, y):
    x, y = np.array(x), np.array(y)
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

# ================= Data collection =================
def collect_data(csv_path, pkl_dir):
    df = pd.read_csv(csv_path)
    S_raw = {}
    S_means = {}
    Gini_vals = {}

    methods_S = ['SP_PLV', 'SP_PLI', 'TP_PLV', 'TP_PLI', 'SP', 'TP', 'SH', 'TH']
    methods_Gini = [f'gini_{m}' for m in methods_S]

    for _, row in df.iterrows():
        band = row['band']
        stage = row['stage']
        pkl_file = row['pkl_file']
        pkl_path = os.path.join(pkl_dir, pkl_file)
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, 'rb') as f:
            detail = pickle.load(f)
        S_dict = detail['S']

        for method in methods_S:
            key = (band, stage, method)
            seq = S_dict[method]
            S_raw.setdefault(key, []).extend(seq)
            S_means.setdefault(key, []).append(np.nanmean(seq))

        for method, gini_col in zip(methods_S, methods_Gini):
            key = (band, stage, method)
            gini = row[gini_col]
            if not np.isnan(gini):
                Gini_vals.setdefault(key, []).append(gini)

    return S_raw, S_means, Gini_vals

# ================= Significance annotation =================
def add_significance_bars(ax, methods, data_dict, comparisons, y_max_start=None, h=0.04):
    raw_pvals = []
    effect_sizes = []
    for m1, m2 in comparisons:
        y1, y2 = data_dict[m1], data_dict[m2]
        _, p = ttest_ind(y1, y2, equal_var=False)
        raw_pvals.append(p)
        effect_sizes.append(abs(cohens_d(y1, y2)))

    _, pvals_corr, _, _ = multipletests(raw_pvals, method='bonferroni')

    if y_max_start is None:
        all_vals = np.concatenate([data_dict[m] for m in methods])
        y_max_start = np.max(all_vals) * 1.2

    y_top = y_max_start
    for k, ((m1, m2), p_corr, d) in enumerate(zip(comparisons, pvals_corr, effect_sizes)):
        i1, i2 = methods.index(m1), methods.index(m2)
        y_line = y_max_start + k * h
        ax.plot([i1, i2], [y_line, y_line], lw=1, c='black')
        star = significance_star(p_corr)
        star_box = TextArea(star, textprops=dict(fontsize=14, fontweight='bold',
                                                 fontfamily='Times New Roman'))
        d_box = TextArea(rf"$d={d:.2f}$", textprops=dict(fontsize=12, fontweight='bold',
                                                         fontfamily='Times New Roman'))
        packed = HPacker(children=[star_box, d_box], align='baseline', pad=0, sep=6)
        ab = AnnotationBbox(packed, ((i1 + i2) / 2, y_line + h / 3),
                            xycoords='data', frameon=False)
        ax.add_artist(ab)
        y_top = max(y_top, y_line + 0.02)

    return y_top

# ================= Half-violin processing =================
def make_left_half_violin(ax, shift=0.1):
    for pc in ax.collections:
        paths = pc.get_paths()
        for path in paths:
            vertices = path.vertices
            x_center = np.mean(vertices[:, 0])
            vertices[:, 0] = np.minimum(vertices[:, 0], x_center)
        pc.set_transform(Affine2D().translate(-shift, 0) + ax.transData)

# ================= S(t) half-violin + boxplot (no significance, supports negative values) =================
def plot_S_halfviolin(data_dict, methods, colors, save_path, comparisons=None, means_dict=None, title=None):
    # Note: comparisons and means_dict parameters are no longer used, kept only for interface compatibility
    df = pd.DataFrame([(m, val) for m in methods for val in data_dict[m]],
                      columns=['Method', 'S'])
    fig, ax = plt.subplots(figsize=(6, 4))

    sns.violinplot(data=df, x='Method', y='S', hue='Method', inner=None, cut=0,
                   linewidth=1.2, palette=colors, legend=False, ax=ax)
    make_left_half_violin(ax, shift=0.08)

    sns.boxplot(data=df, x='Method', y='S', width=0.05,
                showcaps=True, showfliers=False,
                boxprops=dict(facecolor='none', edgecolor='red', linewidth=1),
                whiskerprops=dict(color='red', linewidth=1),
                capprops=dict(color='red', linewidth=1),
                medianprops=dict(color='red', linewidth=1),
                zorder=3, ax=ax)

    for pc in ax.collections:
        pc.set_edgecolor('black')
        pc.set_linewidth(1.2)

    # Use all data to compute nice ticks (automatically adapts to negative values)
    all_vals = np.concatenate([data_dict[m] for m in methods])
    ymin, ymax, yticks = nice_axis_limits(all_vals, max_ticks=5)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(yticks)

    ax.set_ylabel(r'S(t)', fontsize=16)
    ax.set_xlabel(None)
    ax.tick_params(axis='both', labelsize=16, direction='out')
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_bounds(ymin, ymax)
    ax.spines['bottom'].set_bounds(0, len(methods) - 1)

    plt.xticks(fontfamily='Times New Roman')
    plt.yticks(fontfamily='Times New Roman')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ================= Gini bar chart (keep top and right spines, outward ticks, dynamic spacing) =================
def plot_Gini_bar(data_dict, methods, colors, save_path, comparisons=None):
    if comparisons is None:
        comparisons = [(methods[i], methods[j]) for i in range(len(methods))
                       for j in range(i+1, len(methods))]

    means = [np.mean(data_dict[m]) for m in methods]
    stds  = [np.std(data_dict[m])  for m in methods]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(methods, means, yerr=stds, capsize=5, color=colors,
           width=0.4, edgecolor='black', linewidth=1.2, zorder=2)

    for i, m in enumerate(methods):
        y = data_dict[m]
        x = np.random.normal(i, 0.04, size=len(y))
        ax.scatter(x, y, color='white', edgecolor='black', linewidth=0.8,
                   s=20, alpha=1, zorder=3)

    all_vals = np.concatenate([data_dict[m] for m in methods])
    y_base = np.max(all_vals) * 1.5

    # Dynamic spacing: slightly wider for <=3 comparisons, normal for 4-6, compact for more
    n_comps = len(comparisons)
    if n_comps <= 3:
        h = 0.03
    elif n_comps <= 6:
        h = 0.06
    else:
        h = 0.03

    y_top_sig = add_significance_bars(ax, methods, data_dict, comparisons,
                                      y_max_start=y_base, h=h)

    # Determine a reasonable ymax to avoid excessive whitespace
    final_ymax = max(np.max(all_vals) * 1.15, y_top_sig * 1.05)
    all_vals_axis = np.concatenate([all_vals, [final_ymax]])
    ymin, ymax_nice, yticks = nice_axis_limits(all_vals_axis, max_ticks=5)
    if ymax_nice > final_ymax * 1.1:
        ymax = final_ymax
    else:
        ymax = ymax_nice

    ax.set_ylim(ymin, ymax)
    ax.set_yticks(yticks)

    ax.set_ylabel(r'G(x)', fontsize=16)
    ax.set_xlabel(None)
    ax.tick_params(axis='both', labelsize=16, direction='out')
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.xticks(fontfamily='Times New Roman')
    plt.yticks(fontfamily='Times New Roman')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ================= Main program =================
if __name__ == "__main__":
    csv_path = "./results/data/baseline/comparison_plv_pli_spdpe.csv"
    pkl_dir = "./results/data/baseline/details"
    out_dir = "./results/figures/baseline"
    os.makedirs(out_dir, exist_ok=True)

    S_raw, S_means, Gini_vals = collect_data(csv_path, pkl_dir)

    bands = ['Gamma', 'HFO']
    stages = ['pre', 'ictal', 'post']

    colors_phase = ['#698DAF', '#9478AC', '#3E90A3']
    colors_body  = ['#698DAF', '#9478AC', '#3E90A3', '#2170B3']

    phase_methods = ['SP_PLV', 'SP_PLI', 'SP']
    body_methods  = ['SP', 'SH', 'TP', 'TH']

    phase_comps = [('SP_PLV', 'SP_PLI'), ('SP_PLI', 'SP'), ('SP_PLV', 'SP')]
    body_comps  = [('SP', 'SH'), ('SH', 'TP'), ('TP', 'TH'),
                   ('SP', 'TP'), ('SH', 'TH'), ('SP', 'TH')]

    for band in bands:
        for stage in stages:
            # ---------- S(t) phase synchrony ----------
            data_phase_S_raw = {m: S_raw.get((band, stage, m), []) for m in phase_methods}
            data_phase_S_mean = {m: S_means.get((band, stage, m), []) for m in phase_methods}
            if all(len(v) > 0 for v in data_phase_S_raw.values()):
                fname = f'S_violin_phase_{band}_{stage}.png'
                plot_S_halfviolin(data_phase_S_raw, phase_methods, colors_phase,
                                  os.path.join(out_dir, fname),
                                  comparisons=phase_comps,
                                  means_dict=data_phase_S_mean)

            # ---------- Gini phase synchrony ----------
            data_phase_Gini = {m: Gini_vals.get((band, stage, m), []) for m in phase_methods}
            if all(len(v) > 0 for v in data_phase_Gini.values()):
                fname = f'Gini_bar_phase_{band}_{stage}.png'
                plot_Gini_bar(data_phase_Gini, phase_methods, colors_phase,
                              os.path.join(out_dir, fname), comparisons=phase_comps)

            # ---------- S(t) body changes ----------
            data_body_S_raw = {m: S_raw.get((band, stage, m), []) for m in body_methods}
            data_body_S_mean = {m: S_means.get((band, stage, m), []) for m in body_methods}
            if all(len(v) > 0 for v in data_body_S_raw.values()):
                fname = f'S_violin_body_{band}_{stage}.png'
                plot_S_halfviolin(data_body_S_raw, body_methods, colors_body,
                                  os.path.join(out_dir, fname),
                                  comparisons=body_comps,
                                  means_dict=data_body_S_mean)

            # ---------- Gini body changes ----------
            data_body_Gini = {m: Gini_vals.get((band, stage, m), []) for m in body_methods}
            if all(len(v) > 0 for v in data_body_Gini.values()):
                fname = f'Gini_bar_body_{band}_{stage}.png'
                plot_Gini_bar(data_body_Gini, body_methods, colors_body,
                              os.path.join(out_dir, fname), comparisons=body_comps)

    print(f"All figures saved to {out_dir}")