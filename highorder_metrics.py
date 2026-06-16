import os
import pandas as pd
from THFBN import *
from utils import *

# ===================== Parameter configuration =====================
bands = ["Gamma", "HFO"]
band_window = {
    'Gamma': {'win_L': 1, 'step': 0.25},
    'HFO': {'win_L': 0.5, 'step': 0.125},
}
band_K = {'Gamma': 5, 'HFO': 7}

I = 20
sparsity = 0.6
lam = 1e-3
max_windows_per_stage = 60      # Take at most 60 windows per stage

# ===================== Metric computation (F, D, B) =====================
def compute_highorder_metrics(W_th, eps=1e-8):
    F = np.sum(np.maximum(W_th, 0), axis=2)
    D = np.sum(np.abs(np.minimum(W_th, 0)), axis=2)
    B = (F - D) / (F + D + eps)
    return F, D, B

# ===================== Compute F/D/B within a single stage =====================
def process_one_stage_windows(windows, K):
    """
    windows: list of 2D arrays (channels, samples)
    Returns: (F_arr, D_arr, B_arr) with shape (T, N)
    """
    if len(windows) < K + 1:
        return None

    # Limit max number of windows
    if len(windows) > max_windows_per_stage:
        windows = windows[:max_windows_per_stage]

    N = windows[0].shape[0]
    H_states = []
    for X in windows:
        simplicial = simplicial_complex_mvts(X, I, sparsity)
        simplices = simplicial.create_simplicial_complex()
        H, w = build_incidence_matrix(N, simplices)
        h = highorder_state(H, w)
        H_states.append(h)

    W_th = build_temporal_highorder_fcn(H_states, K, lam)
    F_arr, D_arr, B_arr = compute_highorder_metrics(W_th)
    return F_arr, D_arr, B_arr

# ===================== Single seizure-band computation (old code division logic) =====================
def process_one_seizure_band(fpath, row, band):
    pid = int(row.patient)
    sz = int(row.seizure)
    fs = int(row.fs)
    Pre = int(row.Pre)
    Episode = int(row.Episode)
    End = int(row.End)
    K = band_K[band]
    win_L = band_window[band]["win_L"]
    step = band_window[band]["step"]

    # Load full data
    try:
        data = load_data_mat(fpath)         # returns (channels, time)
        if data.ndim != 2:
            raise ValueError("Data is not 2D")
    except Exception as e:
        print(f"Failed to load {fpath}: {e}")
        return []

    N, T_total = data.shape
    total_sec = T_total / fs

    pre_start_sec = 0.0
    pre_end_sec = max(0, Episode - Pre)
    ictal_start_sec = pre_end_sec
    ictal_end_sec = max(0, End - Pre)
    post_start_sec = ictal_end_sec
    post_end_sec = total_sec

    # Ensure stage boundaries are valid
    pre_end_sec = min(pre_end_sec, total_sec)
    ictal_end_sec = min(ictal_end_sec, total_sec)

    stages = {
        "pre": (pre_start_sec, pre_end_sec),
        "ictal": (ictal_start_sec, ictal_end_sec),
        "post": (post_start_sec, post_end_sec)
    }

    results = []
    for stage_name, (t1, t2) in stages.items():
        if t2 <= t1:          # skip if stage duration is zero or negative
            print(f"  P{pid} SZ{sz} {band} {stage_name}: empty interval ({t1:.2f}s - {t2:.2f}s)")
            continue

        # Slice data
        idx1 = int(t1 * fs)
        idx2 = int(t2 * fs)
        if idx1 >= T_total or idx2 <= idx1:
            continue
        data_stage = data[:, idx1:idx2]

        # Sliding windows
        windows = sliding_windows(data_stage, fs, win_L, step)

        # Call core computation
        res = process_one_stage_windows(windows, K)
        if res is None:
            print(f"  P{pid} SZ{sz} {band} {stage_name}: insufficient windows ({len(windows)} windows)")
            continue

        F_arr, D_arr, B_arr = res
        T = F_arr.shape[0]
        for t in range(T):
            for ch_idx in range(F_arr.shape[1]):
                results.append({
                    'Patient': pid,
                    'Seizure': sz,
                    'Band': band,
                    'Stage': stage_name,
                    'Window': t,
                    'Channel': ch_idx,
                    'F': F_arr[t, ch_idx],
                    'D': D_arr[t, ch_idx],
                    'B': B_arr[t, ch_idx]
                })
    return results

# ===================== Load time information table =====================
def load_time_info(timing_xlsx_path):
    df = pd.read_excel(timing_xlsx_path)
    df['Pxxx'] = df['Pxxx'].ffill()                     # Handle merged cells
    df['patient'] = df['Pxxx'].str.extract(r'P(\d+)').astype(int)
    df['seizure'] = df['SZxx'].str.extract(r'SZ(\d+)').astype(int)
    df = df[['patient', 'seizure', 'Pre', 'Episode', 'End', 'fs']]
    return df

# ===================== Load electrode information =====================
def load_electrode_info(elec_xlsx_path):
    df = pd.read_excel(elec_xlsx_path)
    df.columns = ['Patient', 'Channel', 'ROI', 'Zone']  # Adjust column names as needed

    roi_to_lobe = {
        'HIP': 'MTL', 'AMYG': 'MTL',
        'ITG': 'LTL', 'MTG': 'LTL', 'STG': 'LTL', 'PTG': 'LTL',
        'STG.P': 'LTL', 'FFG': 'LTL', 'TP': 'LTL', 'PHG': 'LTL', 'TPO': 'LTL',
        'INS': 'INS', 'INS.A': 'INS', 'INS.M': 'INS', 'INS.P': 'INS', 'INS.G': 'INS',
        'PAR': 'PL', 'PAR.I': 'PL', 'PAR.S': 'PL', 'PAR.SG': 'PL', 'PAR.P': 'PL',
        'OCC': 'OL',
        'MFG': 'FL', 'IFG': 'FL', 'OFG': 'FL', 'SFG': 'FL', 'FP': 'FL',
        'PreCG': 'FL', 'SMA': 'FL', 'ACC': 'FL', 'MCC': 'FL', 'CGS': 'FL', 'CEN': 'FL',
    }

    df['ROI_clean'] = df['ROI'].str.replace(r'\.(L|R)$', '', regex=True)
    df['Lobe'] = df['ROI_clean'].map(roi_to_lobe).fillna('Other')

    df['Laterality'] = None
    df.loc[df['ROI'].str.endswith('.L'), 'Laterality'] = 'Left'
    df.loc[df['ROI'].str.endswith('.R'), 'Laterality'] = 'Right'

    df['Channel'] = df['Channel'] - 1
    df.drop(columns=['ROI_clean'], inplace=True)
    return df

# ===================== Main program =====================
if __name__ == "__main__":
    data_dir = "./data/SEEG"
    timing_xlsx = "./data/SEEG/Patient_Information_Table.xlsx"
    elec_xlsx = "./data/SEEG/Patient_Channel_ROI.xlsx"
    save_dir = "./results/data/FDB_metrics"
    os.makedirs(save_dir, exist_ok=True)

    time_df = load_time_info(timing_xlsx)
    elec_info = load_electrode_info(elec_xlsx)

    cover_mat = elec_info.pivot_table(index='Patient', columns='Lobe', values='Channel', aggfunc='count', fill_value=0)
    cover_mat.to_csv(os.path.join(save_dir, "electrode_coverage_by_lobe.csv"))
    print("Electrode coverage matrix saved.")

    all_results = []
    for _, row in time_df.iterrows():
        pid = int(row.patient)
        sz = int(row.seizure)
        fs = int(row.fs)

        for band in bands:
            if band == "HFO" and fs < 1000:
                continue

            fname = f"P{pid:03d}_SZ{sz:02d}_{band}.mat"
            fpath = os.path.join(data_dir, fname)
            if not os.path.exists(fpath):
                print(f"Missing file: {fpath}")
                continue

            res = process_one_seizure_band(fpath, row, band)
            all_results.extend(res)

    if not all_results:
        print("No data computed. Check .mat file variable names and paths.")
        exit(1)

    df_long = pd.DataFrame(all_results)
    print(f"Computed {len(df_long)} window-level records")

    # Merge electrode information
    df_merged = df_long.merge(elec_info, on=['Patient', 'Channel'], how='left')
    missing = df_merged['ROI'].isna().sum()
    if missing > 0:
        print(f"Warning: {missing} records have no electrode info.")
        df_merged.loc[df_merged['ROI'].isna(),
                      ['ROI', 'Zone', 'Lobe', 'Laterality']] = 'Unknown'

    out_path = os.path.join(save_dir, "channel_level_FDB_windows_all.csv")
    df_merged.to_csv(out_path, index=False)
    print(f"Saved all window-level FDB data with electrode info to {out_path}")
    print("Done.")