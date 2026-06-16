import os
import warnings
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox
from THFBN import *
from utils import *

warnings.filterwarnings("ignore")

# ===================== Parameter configuration =====================
TARGET_PATIENTS = [1, 3, 4, 6, 10]
BANDS = ["Gamma", "HFO"]
BAND_WINDOW = {
    'Gamma': {'win_L': 1, 'step': 0.25},
    'HFO': {'win_L': 0.5, 'step': 0.125},
}
BAND_K = {'Gamma': 5, 'HFO': 7}          # Different bands use different K

I = 20
SPARSITY = 0.6
LAM = 1e-3                              # Ridge regression regularization parameter
MAX_WINDOWS_PER_STAGE = 60
TARGET_WIN_N = 10                       # Number of windows sampled per stage
MAX_CANDIDATE_WINDOWS = 300             # Upper bound for random sampling
N_JOBS = 4
RANDOM_SEED = 42

# ===================== Helper functions =====================
def compute_metrics(y_true, y_pred):
    """NMSE and MAE."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    nmse = np.mean((y_true - y_pred) ** 2) / max(np.mean(y_true ** 2), 1e-12)
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    return nmse, mae

def cosine_similarity(x, y):
    """Cosine similarity between two vectors."""
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)

def pearson_corr(x, y):
    """Pearson correlation coefficient."""
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return np.corrcoef(x, y)[0, 1]

def residual_analysis_local(residuals, max_lag=3):
    """
    Ljung‑Box test for residuals of each node.
    Returns fraction of nodes with white noise (p > 0.05).
    """
    n_nodes = residuals.shape[1]
    white_ps = []
    for i in range(n_nodes):
        res = residuals[:, i]
        if len(res) < 8:
            white_ps.append(np.nan)
            continue
        lag = min(max_lag, len(res) // 2)
        try:
            lb = acorr_ljungbox(res, lags=[lag], return_df=True)
            p = lb["lb_pvalue"].values[0]
        except Exception:
            p = np.nan
        white_ps.append(p)
    white_ps = np.array(white_ps)
    return np.nanmean(white_ps > 0.05)

def evaluate_local_window(H_stack, h_next, lam):
    """
    Fit linear model and kernel ridge model on a sliding window.
    H_stack: shape (N, K+1)   (K = number of past states)
    h_next:  shape (N,)
    lam: regularization parameter for linear model
    """
    N, K_plus_1 = H_stack.shape
    K = K_plus_1 - 1                     # Infer K from H_stack column count

    H_minus = H_stack[:, :K]             # (N, K)
    H_plus  = H_stack[:, 1:]             # (N, K)

    # ---------- Linear model ----------
    W_lin = (
        H_plus
        @ H_minus.T
        @ np.linalg.inv(H_minus @ H_minus.T + lam * np.eye(N))
    )

    y_true_in = H_plus.T                 # (K, N)
    y_pred_in = (W_lin @ H_minus).T      # (K, N)
    nmse_lin_in, mae_lin_in = compute_metrics(y_true_in, y_pred_in)
    residuals = y_true_in - y_pred_in
    frac_white_lin = residual_analysis_local(residuals)

    h_pred_lin = W_lin @ H_stack[:, K]
    nmse_lin_out, mae_lin_out = compute_metrics(h_next, h_pred_lin)
    corr_lin = pearson_corr(h_next, h_pred_lin)
    cos_lin = cosine_similarity(h_next, h_pred_lin)

    # ---------- Kernel ridge (RBF) ----------
    X_train = H_minus.T                  # (K, N)
    y_train = H_plus.T                   # (K, N)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_sc = scaler_X.fit_transform(X_train)
    y_train_sc = scaler_y.fit_transform(y_train)

    kr = KernelRidge(alpha=0.1, gamma=1.0, kernel="rbf")
    kr.fit(X_train_sc, y_train_sc)

    y_pred_kr_sc = kr.predict(X_train_sc)
    y_pred_kr = scaler_y.inverse_transform(y_pred_kr_sc)
    nmse_kr_in, mae_kr_in = compute_metrics(y_train, y_pred_kr)
    residuals_kr = y_train - y_pred_kr
    frac_white_kr = residual_analysis_local(residuals_kr)

    x_last = H_stack[:, K].reshape(1, -1)      # (1, N)
    x_last_sc = scaler_X.transform(x_last)
    h_pred_kr_sc = kr.predict(x_last_sc)
    h_pred_kr = scaler_y.inverse_transform(h_pred_kr_sc).flatten()
    nmse_kr_out, mae_kr_out = compute_metrics(h_next, h_pred_kr)
    corr_kr = pearson_corr(h_next, h_pred_kr)
    cos_kr = cosine_similarity(h_next, h_pred_kr)

    return {
        "nmse_lin_in": nmse_lin_in,
        "mae_lin_in": mae_lin_in,
        "frac_white_lin_in": frac_white_lin,
        "nmse_lin_out": nmse_lin_out,
        "mae_lin_out": mae_lin_out,
        "corr_lin_out": corr_lin,
        "cos_lin_out": cos_lin,
        "nmse_kernel_in": nmse_kr_in,
        "mae_kernel_in": mae_kr_in,
        "frac_white_ker_in": frac_white_kr,
        "nmse_kernel_out": nmse_kr_out,
        "mae_kernel_out": mae_kr_out,
        "corr_kernel_out": corr_kr,
        "cos_kernel_out": cos_kr,
        "delta_nmse_out": nmse_kr_out - nmse_lin_out,
        "delta_corr_out": corr_kr - corr_lin,
    }

def process_stage(data, fs, t1, t2, stage, pid, sz, band, K_val, lam):
    """
    Process a single stage (pre/ictal/post) for a given patient/seizure/band.
    Returns a list of metric dictionaries for each sampled window.
    """
    results = []
    win_L = BAND_WINDOW[band]["win_L"]
    step = BAND_WINDOW[band]["step"]

    # Extract data segment
    data_ = data[:, int(t1 * fs) : int(t2 * fs)]
    windows = sliding_windows(data_, fs, win_L, step)

    if len(windows) < K_val + 2:
        return []

    rng = np.random.default_rng(RANDOM_SEED + pid * 100 + sz)
    candidate = np.arange(len(windows) - K_val - 1)   # Window start index range
    candidate = rng.permutation(candidate)

    valid = 0
    for start in candidate[:MAX_CANDIDATE_WINDOWS]:
        try:
            H_states = []
            # Take K_val+2 consecutive windows: first K_val+1 for H_stack, the (K_val+2)th as prediction target
            for X in windows[start : start + K_val + 2]:
                simplicial = simplicial_complex_mvts(X, I, SPARSITY)
                simplices = simplicial.create_simplicial_complex()
                H, w = build_incidence_matrix(X.shape[0], simplices)
                h = H @ w
                H_states.append(h)

            H_stack = np.stack(H_states[:K_val+1], axis=1)   # (N, K_val+1)
            h_next = H_states[K_val+1]

            res = evaluate_local_window(H_stack, h_next, lam)
            res.update({
                "patient": pid,
                "seizure": sz,
                "band": band,
                "stage": stage,
                "window_start": start,
                "K_used": K_val,
            })
            results.append(res)
            valid += 1
            if valid >= TARGET_WIN_N:
                break
        except Exception as e:
            # Silently skip abnormal windows
            continue

    print(f"P{pid} SZ{sz} {band} {stage}: {valid}")
    return results

def process_task(task):
    """Wrapper for a single (row, band) combination, used by joblib."""
    row, band = task
    pid = int(row.patient)
    sz = int(row.seizure)
    fs = int(row.fs)

    # Select K value based on band
    K_val = BAND_K[band]

    fname = f"P{pid:03d}_SZ{sz:02d}_{band}.mat"
    path = os.path.join(data_dir, fname)
    data = load_data_mat(path)

    Pre = int(row.Pre)
    Episode = int(row.Episode)
    End = int(row.End)
    total_time = data.shape[1] / fs

    stages = {
        "pre": (0, Episode - Pre),
        "ictal": (Episode - Pre, End - Pre),
        "post": (End - Pre, min(End + 20 - Pre, total_time)),
    }

    out = []
    for stage_name, (t1, t2) in stages.items():
        out.extend(process_stage(data, fs, t1, t2, stage_name, pid, sz, band, K_val, LAM))
    return out

if __name__ == "__main__":
    data_dir = "./data/SEEG"
    save_dir = "./results/data/nonlinear"
    os.makedirs(save_dir, exist_ok=True)
    excel_path = "./data/SEEG/Patient_Information_Table.xlsx"

    df = pd.read_excel(excel_path, sheet_name="ictal")
    df["Pxxx"] = df["Pxxx"].ffill()
    df["patient"] = df["Pxxx"].str.extract(r"P(\d+)").astype(int)
    df["seizure"] = df["SZxx"].str.extract(r"SZ(\d+)").astype(int)
    df = df[df["patient"].isin(TARGET_PATIENTS)]
    df = df.sort_values(["patient", "seizure"])

    tasks = []
    for band in BANDS:
        for _, row in df.iterrows():
            fs = int(row.fs)
            if band == "HFO" and fs < 1000:
                continue
            tasks.append((row, band))

    results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=10)(
        delayed(process_task)(t) for t in tasks
    )

    all_results = [x for sub in results for x in sub]
    df_out = pd.DataFrame(all_results)
    save_csv = os.path.join(save_dir, "local_fit_comparison.csv")
    df_out.to_csv(save_csv, index=False)

    print("\n✅ Done.")
    print(f"Total windows: {len(df_out)}")