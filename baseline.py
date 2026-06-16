import os
import pickle
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import hilbert
from THFBN import *
from utils import *

# ===================== Parameter configuration =====================
target_patients = [1, 3, 4, 6, 10]
bands = ["Gamma", "HFO"]
band_window = {
    'Gamma': {'win_L': 1, 'step': 0.25},
    'HFO': {'win_L': 0.5, 'step': 0.125},
}
I = 20
sparsity = 0.6
lam = 1e-3
n_surrogates = 100
n_jobs = 4
max_windows_per_stage = 60

# ===================== Static pairwise network construction =====================
def build_static_pairwise_plv(ts_all):
    """PLV (Phase Locking Value)"""
    N, T = ts_all.shape
    analytic = hilbert(ts_all, axis=1)
    phase = np.angle(analytic)
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            plv = np.abs(np.mean(np.exp(1j * (phase[i] - phase[j]))))
            W[i, j] = plv
            W[j, i] = plv
    return W

def build_static_pairwise_pli(ts_all):
    """PLI (Phase Lag Index)"""
    N, T = ts_all.shape
    analytic = hilbert(ts_all, axis=1)
    phase = np.angle(analytic)
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            phase_diff = phase[i] - phase[j]
            pli = np.abs(np.mean(np.sign(np.sin(phase_diff))))
            W[i, j] = pli
            W[j, i] = pli
    return W

def build_pairwise_from_simplices(simplices, N):
    """
    Directly extract 1-simplices (edges) from the simplex list returned by create_simplicial_complex()
    simplices: [(nodes, weight), ...]  nodes is a list or tuple
    """
    W = np.zeros((N, N))
    for nodes, weight in simplices:
        if len(nodes) == 2:           # take only pairwise relationships
            i, j = nodes[0], nodes[1]
            W[i, j] += weight
            W[j, i] += weight
    max_val = np.max(W)
    if max_val > 0:
        W /= max_val
    return W

# ===================== Temporal network construction =====================
def adjacency_to_edge_vector(W):
    iu = np.triu_indices_from(W, k=1)
    return W[iu]

def edge_vector_to_adjacency(vec, N):
    W = np.zeros((N, N))
    iu = np.triu_indices(N, k=1)
    W[iu] = vec
    W.T[iu] = vec
    return W

def build_temporal_pairwise_fcn(W_static_list, K=5, lam=1e-3):
    edge_states = np.array([adjacency_to_edge_vector(W) for W in W_static_list])
    T, M = edge_states.shape
    N = W_static_list[0].shape[0]
    Ws = []
    for k in range(T - K):
        P_win = edge_states[k:k+K+1]
        P_minus = P_win[:-1].T
        P_plus  = P_win[1:].T
        A = P_plus @ P_minus.T @ np.linalg.inv(P_minus @ P_minus.T + lam * np.eye(M))
        p_last = P_win[-2]
        p_pred = A @ p_last
        W_pred = edge_vector_to_adjacency(p_pred, N)
        Ws.append(W_pred)
    return Ws

def build_temporal_highorder_fcn(H_states, K=5, lam=1e-3):
    Ws = []
    T = len(H_states)
    for k in range(T - K):
        H_win = H_states[k:k+K+1]
        H_minus = np.stack(H_win[:-1], axis=1)  # N × K
        H_plus  = np.stack(H_win[1:], axis=1)   # N × K
        N = H_minus.shape[0]
        W = H_plus @ H_minus.T @ np.linalg.inv(H_minus @ H_minus.T + lam * np.eye(N))
        Ws.append(W)
    return Ws

def build_static_highorder_fcn(N, simplices):
    """Static high-order network (SH), accepts simplices list"""
    H, weights = build_incidence_matrix(N, simplices)
    h = highorder_state(H, weights)
    W = np.outer(h, h)
    return W

# ===================== Evaluation metrics =====================
def adjacency_similarity(W1, W2):
    iu = np.triu_indices_from(W1, k=1)
    a = W1[iu]
    b = W2[iu]
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0
    return np.corrcoef(a, b)[0, 1]

def compute_S_t(W_list):
    return np.array([adjacency_similarity(W_list[t], W_list[t+1]) for t in range(len(W_list)-1)])

def gini_coefficient(x):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    x = x[x >= 0]
    if len(x) == 0 or np.sum(x) == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    idx = np.arange(1, n+1)
    return (2 * np.sum(idx * x)) / (n * np.sum(x)) - (n+1)/n

# ===================== Surrogate data =====================
def phase_randomize(ts):
    T = len(ts)
    fft = np.fft.rfft(ts)
    mag = np.abs(fft)
    phase = np.angle(fft)
    random_phase = np.random.uniform(0, 2*np.pi, len(fft))
    random_phase[0] = phase[0]
    if T % 2 == 0:
        random_phase[-1] = phase[-1]
    fft_new = mag * np.exp(1j * random_phase)
    return np.fft.irfft(fft_new, n=T)

def calc_p_value(true_val, surr_vals):
    surr_vals = np.array(surr_vals)
    return (np.sum(surr_vals >= true_val) + 1) / (len(surr_vals) + 1)

# ===================== Single-stage processing =====================
def process_one_stage(data, fs, t_start, t_end, stage_name, pid, sz, band, win_L, step, K_val):
    data_ = data[:, int(t_start*fs):int(t_end*fs)]
    windows = sliding_windows(data_, fs, win_L, step)
    if len(windows) < K_val + 2:
        return None, None

    if len(windows) > max_windows_per_stage:
        windows = windows[:max_windows_per_stage]

    N = data.shape[0]
    W_plv_list = []
    W_pli_list = []
    W_sp_list = []
    W_sh_list = []
    H_states = []

    for X in windows:
        W_plv_list.append(build_static_pairwise_plv(X))
        W_pli_list.append(build_static_pairwise_pli(X))

        simplicial = simplicial_complex_mvts(X, I, sparsity)
        simplices = simplicial.create_simplicial_complex()
        H, w = build_incidence_matrix(N, simplices)
        h = highorder_state(H, w)

        W_sp_list.append(build_pairwise_from_simplices(simplices, N))
        H_states.append(h)
        W_sh_list.append(build_static_highorder_fcn(N, simplices))

    # Temporal networks
    W_tp_plv = build_temporal_pairwise_fcn(W_plv_list, K_val, lam)
    W_tp_pli = build_temporal_pairwise_fcn(W_pli_list, K_val, lam)
    W_tp_sp  = build_temporal_pairwise_fcn(W_sp_list, K_val, lam)
    W_th     = build_temporal_highorder_fcn(H_states, K_val, lam)

    T_eff = len(windows) - K_val

    S_plv    = compute_S_t(W_plv_list[:T_eff])
    S_pli    = compute_S_t(W_pli_list[:T_eff])
    S_sp     = compute_S_t(W_sp_list[:T_eff])
    S_sh     = compute_S_t(W_sh_list[:T_eff])
    S_tp_plv = compute_S_t(W_tp_plv)
    S_tp_pli = compute_S_t(W_tp_pli)
    S_tp_sp  = compute_S_t(W_tp_sp)
    S_th     = compute_S_t(W_th)

    gini = {
        'SP_PLV': gini_coefficient(S_plv),
        'SP_PLI': gini_coefficient(S_pli),
        'TP_PLV': gini_coefficient(S_tp_plv),
        'TP_PLI': gini_coefficient(S_tp_pli),
        'SP':     gini_coefficient(S_sp),
        'TP':     gini_coefficient(S_tp_sp),
        'SH':     gini_coefficient(S_sh),
        'TH':     gini_coefficient(S_th),
    }

    # ========== Surrogate data test: generate one by one to avoid memory explosion ==========
    surr_gini = {key: [] for key in gini.keys()}

    for _ in range(n_surrogates):
        # Generate a single surrogate (phase randomization)
        surr = np.zeros_like(data_)
        for ch in range(data_.shape[0]):
            surr[ch] = phase_randomize(data_[ch])

        ws = sliding_windows(surr, fs, win_L, step)
        if len(ws) < K_val + 2:
            continue
        if len(ws) > max_windows_per_stage:
            ws = ws[:max_windows_per_stage]

        W_plv_s, W_pli_s, W_sp_s, W_sh_s, H_s = [], [], [], [], []
        for Xs in ws:
            W_plv_s.append(build_static_pairwise_plv(Xs))
            W_pli_s.append(build_static_pairwise_pli(Xs))
            simplicial = simplicial_complex_mvts(Xs, I, sparsity)
            simplices = simplicial.create_simplicial_complex()
            H, w = build_incidence_matrix(N, simplices)
            h = highorder_state(H, w)
            W_sp_s.append(build_pairwise_from_simplices(simplices, N))
            H_s.append(h)
            W_sh_s.append(build_static_highorder_fcn(N, simplices))

        W_tp_plv_s = build_temporal_pairwise_fcn(W_plv_s, K_val, lam)
        W_tp_pli_s = build_temporal_pairwise_fcn(W_pli_s, K_val, lam)
        W_tp_sp_s  = build_temporal_pairwise_fcn(W_sp_s, K_val, lam)
        W_th_s     = build_temporal_highorder_fcn(H_s, K_val, lam)

        surr_gini['SP_PLV'].append(gini_coefficient(compute_S_t(W_plv_s[:T_eff])))
        surr_gini['SP_PLI'].append(gini_coefficient(compute_S_t(W_pli_s[:T_eff])))
        surr_gini['TP_PLV'].append(gini_coefficient(compute_S_t(W_tp_plv_s)))
        surr_gini['TP_PLI'].append(gini_coefficient(compute_S_t(W_tp_pli_s)))
        surr_gini['SP'].append(gini_coefficient(compute_S_t(W_sp_s[:T_eff])))
        surr_gini['TP'].append(gini_coefficient(compute_S_t(W_tp_sp_s)))
        surr_gini['SH'].append(gini_coefficient(compute_S_t(W_sh_s[:T_eff])))
        surr_gini['TH'].append(gini_coefficient(compute_S_t(W_th_s)))

    p_vals = {key: calc_p_value(gini[key], surr_gini[key]) for key in gini.keys()}
    # ========================================================

    detail = {
        "S": {
            "SP_PLV": S_plv,
            "SP_PLI": S_pli,
            "TP_PLV": S_tp_plv,
            "TP_PLI": S_tp_pli,
            "SP": S_sp,
            "TP": S_tp_sp,
            "SH": S_sh,
            "TH": S_th,
        },
        "Gini": gini,
        "p_value": p_vals,
    }

    flat = {
        "patient": pid,
        "seizure": sz,
        "band": band,
        "stage": stage_name,
    }
    for key in gini.keys():
        flat[f"gini_{key}"] = gini[key]
        flat[f"p_{key}"] = p_vals[key]

    return flat, detail

# ===================== Batch processing entry point =====================
def process_one_seizure_band(row, band):
    pid = int(row.patient)
    sz = int(row.seizure)
    fs = int(row.fs)
    if band == "HFO" and fs < 1000:
        return []
    fname = f"P{pid:03d}_SZ{sz:02d}_{band}.mat"
    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
        print(f"File not found: {fpath}")
        return []
    data = load_data_mat(fpath)
    Pre = int(row.Pre)
    Episode = int(row.Episode)
    End = int(row.End)
    total_time = data.shape[1] / fs
    win_L = band_window[band]["win_L"]
    step = band_window[band]["step"]

    # Gamma uses K=5, HFO uses K=7
    band_K = {'Gamma': 5, 'HFO': 7}
    K_val = band_K[band]

    stages = {
        "pre":   (Pre, Episode - Pre),
        "ictal": (Episode - Pre, End - Pre),
        "post":  (End - Pre, min(End - Pre + 20, total_time))
    }
    pkl_dir = os.path.join(save_dir, "details")
    os.makedirs(pkl_dir, exist_ok=True)

    csv_rows = []
    for stage_name, (t1, t2) in stages.items():
        if t2 - t1 < win_L:
            continue
        flat, detail = process_one_stage(data, fs, t1, t2, stage_name, pid, sz, band, win_L, step, K_val)
        if flat is None:
            continue

        pkl_name = f"P{pid:03d}_SZ{sz:02d}_{band}_{stage_name}.pkl"
        pkl_path = os.path.join(pkl_dir, pkl_name)
        with open(pkl_path, "wb") as f:
            pickle.dump(detail, f)

        flat["pkl_file"] = pkl_name
        csv_rows.append(flat)

    return csv_rows

if __name__ == "__main__":
    data_dir = "./data/SEEG"
    save_dir = "./results/data/baseline"
    os.makedirs(save_dir, exist_ok=True)

    excel_path = "./data/SEEG/Patient_Information_Table.xlsx"
    df = pd.read_excel(excel_path, sheet_name="ictal")
    df["Pxxx"] = df["Pxxx"].ffill()
    df["patient"] = df["Pxxx"].str.extract(r"P(\d+)").astype(int)
    df["seizure"] = df["SZxx"].str.extract(r"SZ(\d+)").astype(int)
    df = df[df["patient"].isin(target_patients)]
    df = df.sort_values(["patient", "seizure"])

    tasks = []
    for _, row in df.iterrows():
        for band in bands:
            tasks.append((row, band))

    all_rows = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_one_seizure_band)(row, band) for (row, band) in tasks
    )

    flat_results = [item for sublist in all_rows for item in sublist]
    df_out = pd.DataFrame(flat_results)

    out_csv = os.path.join(save_dir, "comparison_plv_pli_spdpe.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"Saved {len(df_out)} rows to {out_csv}")
    print(f"Detail pickle files saved to {os.path.join(save_dir, 'details')}")