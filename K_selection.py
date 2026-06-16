import os
import pandas as pd
import pickle
from joblib import Parallel, delayed
from THFBN import *

# ================= Parameters =================
I = 20
sparsity = 0.6
target_patients = [1, 3, 4, 6, 10]
bands = ["Gamma", "HFO"]
band_window = {
    'Gamma': {'win_L': 1, 'step': 0.25},
    'HFO': {'win_L': 0.5, 'step': 0.125},
}
K_list = [2, 3, 4, 5, 6, 7]
max_windows = 60
lam = 1e-3
n_jobs = 6

# ================= Core computation functions =================
def process_single_window_state(X, I, sparsity):
    """
    X : ndarray of shape (samples, channels)
    Returns high-order state vector
    """
    n_samples, n_channels = X.shape
    if n_channels < 3:
        return np.zeros(n_channels)

    # simplicial_complex_mvts expects input shape (channels, samples)
    simplicial = simplicial_complex_mvts(X.T, I, sparsity)
    simplices = simplicial.create_simplicial_complex()
    H, w = build_incidence_matrix(n_channels, simplices)
    return highorder_state(H, w)

def relative_stability(W_list):
    diffs = []
    for i in range(len(W_list)-1):
        num = np.linalg.norm(W_list[i+1] - W_list[i], 'fro')
        den = np.linalg.norm(W_list[i], 'fro') + 1e-8
        diffs.append(num / den)
    return np.mean(diffs), np.std(diffs)

def hub_consistency(Ws):
    strengths = np.array([np.sum(np.abs(W), axis=1) for W in Ws])
    valid_mask = np.std(strengths, axis=0) > 1e-12
    if not np.any(valid_mask):
        return np.nan
    strengths_valid = strengths[:, valid_mask]
    corr_mat = np.corrcoef(strengths_valid.T)
    triu_idx = np.triu_indices_from(corr_mat, k=1)
    return np.nanmean(corr_mat[triu_idx])

# ================= Single task processor =================
def process_one_task(task):
    (pid, sz, band, data_dir, win_L, step,
     I, sparsity, K_list, max_windows,
     Pre, Episode, End, fs) = task

    fname = f"P{pid:03d}_SZ{sz:02d}_{band}.mat"
    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
        return None

    try:
        data = load_data_mat(fpath)
    except Exception as e:
        print(f"Load error {fname}: {e}")
        return None

    # Extract ictal data
    start_idx = int((Episode - Pre) * fs)
    end_idx = int((End - Pre) * fs)
    data_ = data[:, start_idx:end_idx].T   # Transpose to (samples, channels)

    win_len_samples = int(win_L * fs)
    step_samples = int(step * fs)
    n_samples, n_channels = data_.shape

    state_vectors = []
    start = 0
    win_idx = 0
    while start + win_len_samples <= n_samples and win_idx < max_windows:
        X = data_[start:start+win_len_samples, :]   # (win_samples, channels)
        h = process_single_window_state(X, I, sparsity)
        state_vectors.append(h)
        start += step_samples
        win_idx += 1

    if len(state_vectors) < max(K_list) + 1:
        return None

    results = {}
    for K in K_list:
        Ws = build_temporal_highorder_fcn(state_vectors, K, lam=lam)
        if len(Ws) < 2:
            continue
        stab_mean, stab_std = relative_stability(Ws)
        hub_corr = hub_consistency(Ws)
        results[K] = {
            'stability_mean': stab_mean,
            'stability_std': stab_std,
            'hub_consistency': hub_corr,
            'num_Ws': len(Ws)
        }
    return (pid, sz, band, results)

# ================= Main =================
if __name__ == "__main__":
    data_dir = "./data/SEEG"
    save_dir = "./results/data/ablation"
    excel_path = "./data/SEEG/Patient_Information_Table.xlsx"
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_excel(excel_path, sheet_name="ictal")
    df['Pxxx'] = df['Pxxx'].ffill()
    df['patient'] = df['Pxxx'].str.extract(r'P(\d+)').astype(int)
    df['seizure'] = df['SZxx'].str.extract(r'SZ(\d+)').astype(int)
    df = df.dropna(subset=['patient', 'seizure'])
    df = df[df['patient'].isin(target_patients)]
    df = df.sort_values(['patient', 'seizure'])

    tasks = []
    for pid in target_patients:
        sub = df[df['patient'] == pid]
        for _, row in sub.iterrows():
            sz = int(row['seizure'])
            fs = int(row['fs'])
            for band in bands:
                if band == "HFO" and fs < 1000:
                    continue
                win_L = band_window[band]['win_L']
                step = band_window[band]['step']
                tasks.append((
                    pid, sz, band, data_dir, win_L, step,
                    I, sparsity, K_list, max_windows,
                    row['Pre'], row['Episode'], row['End'], fs
                ))

    res_list = Parallel(n_jobs=n_jobs, verbose=10)(delayed(process_one_task)(task) for task in tasks)

    for res in res_list:
        if res is None:
            continue
        pid, sz, band, results = res
        save_path = os.path.join(save_dir, f"P{pid:03d}_SZ{sz:02d}_{band}_K_ablation_THFBN.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
    print("All K ablation tasks completed.")