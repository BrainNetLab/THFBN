import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel  # Changed to paired t-test

# ===================== Paths and parameters =====================
CSV_PATH = "./results/data/FDB_metrics/channel_level_FDB_windows_all.csv"
BASE_OUT_DIR = "./results/figures"
GROUPED_DIR = os.path.join(BASE_OUT_DIR, "highorder_metrics_lobe_box_grouped")
os.makedirs(GROUPED_DIR, exist_ok=True)
df = pd.read_csv(CSV_PATH)

# --------------------- Only keep 5 brain lobes ---------------------
selected_lobes = ['FL', 'INS', 'MTL', 'LTL', 'PL']
df = df[df['Lobe'].isin(selected_lobes)].copy()
lobe_order = selected_lobes

# Stage color palette
stage_palette = {'pre': '#3E90A3', 'ictal': '#9478AC', 'post': '#2170B3'}

df['Lobe'] = pd.Categorical(df['Lobe'], categories=lobe_order, ordered=True)
stage_order = ['pre', 'ictal', 'post']
band_order = ['Gamma', 'HFO']
metric_list = ['F', 'D', 'B']
df['Stage'] = pd.Categorical(df['Stage'], categories=stage_order, ordered=True)
df['Band'] = pd.Categorical(df['Band'], categories=band_order, ordered=True)

# ===================== Aggregate by seizure =====================
agg_df = df.groupby(['Patient', 'Seizure', 'Band', 'Stage', 'Lobe'],
                    observed=False)[metric_list].mean().reset_index()

# ===================== Helper functions =====================
def nice_axis_limits(all_data, max_ticks=4):
    vmin = np.nanmin(all_data)
    vmax = np.nanmax(all_data)
    if vmax == vmin:
        return vmin - 0.5, vmax + 0.5, np.array([vmin, vmax]), 1.0
    raw_step = (vmax - vmin) / (max_ticks - 1)
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
    if vmin >= 0:
        ymin = 0.0
    else:
        ymin = np.floor(vmin / step) * step
    ymax = np.ceil(vmax / step) * step
    yticks = np.arange(ymin, ymax + step / 2, step)
    yticks = yticks[(yticks >= ymin) & (yticks <= ymax)]
    return ymin, ymax, yticks, step

def cohens_d_paired(x, y):
    diff = np.asarray(x) - np.asarray(y)
    sd = diff.std(ddof=1)
    if sd == 0:
        return np.nan
    return diff.mean() / sd

def fmt_pvalue(p, threshold=0.001):
    if pd.isna(p):
        return "NA"
    if p < threshold:
        return f"<{threshold}"
    return f"{p:.3f}"

def bonferroni_correction(p_values):
    p = np.asarray(p_values, dtype=float)
    corrected = np.full_like(p, np.nan)

    valid = ~np.isnan(p)
    m = valid.sum()

    corrected[valid] = np.minimum(p[valid] * m, 1.0)
    return corrected

# ===================== Global style =====================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

# ===================== Grouped boxplot generation (band × metric) =====================
print("Starting grouped boxplot generation...")
for band in band_order:
    for metric in metric_list:
        sub = agg_df[(agg_df['Band'] == band)].dropna(subset=[metric])
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 2.5))

        sns.boxplot(
            data=sub, x='Lobe', y=metric, hue='Stage',
            order=lobe_order, hue_order=stage_order,
            palette=stage_palette,
            width=0.8, gap=0.3, linewidth=0,
            fliersize=0, legend=False, ax=ax,
            showcaps=False,
            boxprops=dict(facecolor='white', edgecolor='black', linewidth=1),
            whiskerprops=dict(color='black', linewidth=1),
            medianprops=dict(color='black', linewidth=1),
            capprops=dict(color='black', linewidth=0)
        )

        sns.stripplot(
            data=sub, x='Lobe', y=metric, hue='Stage',
            order=lobe_order, hue_order=stage_order,
            palette=stage_palette,
            dodge=True,
            jitter=0.15,
            size=4,
            edgecolor='white',
            linewidth=0.5,
            alpha=0.8,
            legend=False, ax=ax,
            zorder=4
        )

        y_data = sub[metric].values
        if metric == 'B':
            ymin, ymax, yticks, step = nice_axis_limits(y_data, max_ticks=5)
        else:
            ymin, ymax, yticks, step = nice_axis_limits(y_data, max_ticks=4)
        ymin_extend = ymin - step * 0.1

        ax.spines['bottom'].set_position(('data', ymin_extend))
        ax.spines['left'].set_position(('outward', 8))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        ax.spines['left'].set_bounds(ymin, ymax)
        ax.spines['bottom'].set_bounds(0, len(lobe_order) - 1)
        ax.set_ylim(ymin_extend, ymax)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{y:g}" for y in yticks])

        ax.tick_params(axis='x', which='both', bottom=True, top=False,
                       direction='out', length=5, width=1.2, pad=8)
        ax.tick_params(axis='y', direction='out', length=5, width=1.2)

        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.tight_layout()
        fname = f"{band}_{metric}_grouped_box.png"
        plt.savefig(os.path.join(GROUPED_DIR, fname), facecolor='white')
        plt.close()
        print(f"Grouped plot saved: {fname}")

# ===================== Lobe patient count statistics table =====================
stats_df = df[df['Lobe'].isin(selected_lobes)].copy()
total_patients = stats_df['Patient'].nunique()
lobe_patient_count = stats_df.groupby('Lobe')['Patient'].nunique()
lobe_patient_count = lobe_patient_count.reindex(selected_lobes).fillna(0).astype(int)
lobe_patient_pct = (lobe_patient_count / total_patients * 100).round(0).astype(int)

summary_tbl = pd.DataFrame({
    'Lobe': selected_lobes,
    'Patients': lobe_patient_count.values,
    'Percentage': lobe_patient_pct.values
})
summary_tbl['Patients (%)'] = summary_tbl.apply(
    lambda row: f"{row['Patients']} ({row['Percentage']}%)", axis=1
)
summary_tbl = summary_tbl[['Lobe', 'Patients (%)']]

csv_out = os.path.join(GROUPED_DIR, "lobe_patient_stats.csv")
summary_tbl.to_csv(csv_out, index=False, encoding='utf-8-sig')
print(f"\nLobe patient statistics table saved to: {csv_out}")
print(summary_tbl.to_string(index=False))

# ===================== Supplementary Table: paired t-test + Bonferroni =====================
print("\nCalculating paired t-test grouped by lobe (with Bonferroni correction)...")

rows = []
for band in band_order:
    band_data = agg_df[agg_df['Band'] == band]
    for metric in metric_list:
        # First collect all raw p-values for this family of comparisons
        family_pvals = []  # (lobe, s1, s2, p_val)
        for lobe in lobe_order:
            lobe_data = band_data[band_data['Lobe'] == lobe]
            if lobe_data.empty:
                continue
            # Pivot so that pre/ictal/post for the same seizure are aligned
            tmp = lobe_data.pivot_table(index=['Patient','Seizure'], columns='Stage', values=metric)
            # Three comparison pairs
            for s1, s2 in [('pre','ictal'), ('ictal','post'), ('pre','post')]:
                pair = tmp[[s1, s2]].dropna()
                if len(pair) >= 2:
                    _, p_val = ttest_rel(pair[s1], pair[s2])
                else:
                    p_val = np.nan
                family_pvals.append((lobe, s1, s2, p_val))

        # Bonferroni correction
        p_vals_only = [item[3] for item in family_pvals]
        corrected = bonferroni_correction(p_vals_only)
        correction_map = {(item[0], item[1], item[2]): corr
                          for item, corr in zip(family_pvals, corrected)}

        # Generate table rows
        for lobe in lobe_order:
            lobe_data = band_data[band_data['Lobe'] == lobe]
            if lobe_data.empty:
                continue
            tmp = lobe_data.pivot_table(index=['Patient','Seizure'], columns='Stage', values=metric)
            # Full values for three stages for mean ± SD
            pre_vals = tmp['pre'].dropna()
            ictal_vals = tmp['ictal'].dropna()
            post_vals = tmp['post'].dropna()

            def fmt_mean_sd(series):
                if len(series) == 0:
                    return "NA"
                return f"{series.mean():.3f} +/- {series.std(ddof=1):.3f}"

            pre_str = fmt_mean_sd(pre_vals)
            ictal_str = fmt_mean_sd(ictal_vals)
            post_str = fmt_mean_sd(post_vals)

            for s1, s2 in [('pre','ictal'), ('ictal','post'), ('pre','post')]:
                pair = tmp[[s1, s2]].dropna()
                n_pairs = len(pair)
                if n_pairs >= 2:
                    _, p_val = ttest_rel(pair[s1], pair[s2])
                    d_val = cohens_d_paired(pair[s1], pair[s2])
                else:
                    p_val = np.nan
                    d_val = np.nan
                p_corr = correction_map.get((lobe, s1, s2), np.nan)

                rows.append({
                    'Band': band,
                    'Metric': metric,
                    'Lobe': lobe,
                    'Comparison': f"{s1.capitalize()} vs {s2.capitalize()}",
                    'N_pairs': n_pairs,
                    'Pre (mean +/- SD)': pre_str,
                    'Ictal (mean +/- SD)': ictal_str,
                    'Post (mean +/- SD)': post_str,
                    'Cohens_d': f"{d_val:.3f}" if pd.notna(d_val) else "NA",
                    'p_value': fmt_pvalue(p_val),
                    'p_corr': fmt_pvalue(p_corr),
                })

supp_df = pd.DataFrame(rows)
supp_df = supp_df[['Band', 'Metric', 'Lobe', 'Comparison', 'N_pairs',
                   'Pre (mean +/- SD)', 'Ictal (mean +/- SD)', 'Post (mean +/- SD)',
                   'Cohens_d', 'p_value', 'p_corr']]

supp_csv = os.path.join(GROUPED_DIR, "supplementary_paired_comparisons_by_lobe.csv")
supp_df.to_csv(supp_csv, index=False, encoding='utf-8-sig')
print(f"\nSupplementary Table (Paired t-test, Bonferroni) saved to: {supp_csv}")
print(supp_df.to_string(index=False))

print("\nAll figures generated!")