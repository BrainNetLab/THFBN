from construct_simplicial_complexes import *
from utils import *
from itertools import combinations
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import networkx as nx
import os

# ================= Adjustable parameters =================
I_list = [16, 20, 24]
sparsity_list = [0.5, 0.6, 0.7, 0.8]
target_patients = [1, 3, 4, 6, 10]
bands = ["Gamma", "HFO"]
band_window = {
    'Gamma': {'win_L': 1, 'step': 0.25},
    'HFO': {'win_L': 0.5, 'step': 0.125},
}

target_win_N = 10          # The number of valid windows that need to be collected for each parameter combination
max_total_windows = 500    # The maximum number of windows that a single task can attempt (to prevent infinite loops)
n_surrogate = 100
N_JOBS = 6
base_random_seed = 42

# ================= Closed triangle counting =================
def count_closed_triplets(edge_set, triplet_list):
    """
    Count how many candidate triplets are fully closed in edge_set
    """
    cnt = 0
    for i, j, k in triplet_list:
        if (i, j) in edge_set and (i, k) in edge_set and (j, k) in edge_set:
            cnt += 1
    return cnt

# ================= Single-window processing =================
def process_single_window(X, I, sparsity_list, pid, sz, band, win_idx, n_surrogate, base_seed):
    """
    X : shape (n_samples, n_channels)
    Returns a list of results for this window across all sparsity levels
    """
    assert X.ndim == 2, f"X must be 2D, got {X.ndim}D"
    n_samples, n_channels = X.shape

    if n_channels < 3:
        return []

    # 1. Precompute all edge SPDPE values
    edges = []
    edge_weights = []
    for i, j in combinations(range(n_channels), 2):
        w = SPDPE(X[:, [i, j]], I)
        edges.append((i, j))
        edge_weights.append(w)
    edge_weights = np.array(edge_weights)

    # 2. Precompute all triplet SPDPE values
    triplets = []
    tri_weights = []
    for i, j, k in combinations(range(n_channels), 3):
        w = SPDPE(X[:, [i, j, k]], I)
        triplets.append((i, j, k))
        tri_weights.append(w)
    tri_weights = np.array(tri_weights)

    results = []
    for sparsity in sparsity_list:
        if len(edge_weights) == 0:
            continue

        # Edge thresholding
        th_edge = np.quantile(edge_weights, sparsity)
        real_edges = [e for e, w in zip(edges, edge_weights) if w >= th_edge]
        edge_set = set(real_edges)
        n_edges = len(edge_set)
        if n_edges < 2:
            continue

        # Triplet thresholding
        th_tri = np.quantile(tri_weights, sparsity)
        valid_triplets = [t for t, w in zip(triplets, tri_weights) if w >= th_tri]
        n_triplets = len(valid_triplets)
        if n_triplets == 0:
            continue

        # Real closed count and RHO
        real_closed = count_closed_triplets(edge_set, valid_triplets)
        RHO_real = real_closed / n_triplets

        # Build the original graph
        G = nx.Graph()
        G.add_nodes_from(range(n_channels))
        G.add_edges_from(real_edges)

        n_nodes = n_channels
        max_edges = n_nodes * (n_nodes - 1) / 2
        density = n_edges / max_edges
        degrees = np.array([d for _, d in G.degree()])
        mean_degree = np.mean(degrees)
        std_degree = np.std(degrees)

        null_closed_list = []
        null_rho_list = []
        swap_change_list = []
        nswap = max(100, n_edges * 10)

        # ---------- Generate randomized samples ----------
        for surr_idx in range(n_surrogate):
            seed_sequence = np.random.SeedSequence([base_seed, pid, sz, win_idx, surr_idx])
            seed = int(seed_sequence.generate_state(1)[0])

            G_rand = G.copy()
            try:
                nx.double_edge_swap(G_rand, nswap=nswap, max_tries=50 * nswap, seed=seed)
            except (nx.NetworkXError, nx.NetworkXAlgorithmError):
                continue

            # Swap succeeded, record this surrogate
            rand_edges = list(G_rand.edges())
            real_edge_set = set(tuple(sorted(e)) for e in real_edges)
            rand_edge_set = set(tuple(sorted(e)) for e in rand_edges)
            edge_change_ratio = len(real_edge_set.symmetric_difference(rand_edge_set)) / (2 * n_edges)
            swap_change_list.append(edge_change_ratio)

            rand_set = set(tuple(sorted(e)) for e in rand_edges)
            closed = count_closed_triplets(rand_set, valid_triplets)
            null_closed_list.append(closed)
            null_rho_list.append(closed / n_triplets)

        n_success = len(null_closed_list)
        swap_success_rate = n_success / n_surrogate

        # If no surrogate succeeded, set result as invalid (swap_success_rate=0, p=NaN)
        if n_success == 0:
            results.append({
                "patient": pid,
                "seizure": sz,
                "band": band,
                "I": I,
                "sparsity": sparsity,
                "win_idx": win_idx,
                "RHO_real": RHO_real,
                "closed_real": real_closed,
                "closed_null_mean": np.nan,
                "RHO_null": np.nan,
                "Z": np.nan,
                "p": np.nan,
                "n_channels": n_nodes,
                "n_edges": n_edges,
                "n_triplets_cand": n_triplets,
                "density": density,
                "mean_degree": mean_degree,
                "std_degree": std_degree,
                "swap_efficiency": np.nan,
                "swap_success_rate": 0.0
            })
            continue

        null_closed_arr = np.array(null_closed_list)
        null_rho_arr = np.array(null_rho_list)

        swap_efficiency = np.mean(swap_change_list)
        closed_null_mean = np.mean(null_closed_arr)
        RHO_null = np.mean(null_rho_arr)
        eps = 1e-4
        sigma = max(np.std(null_rho_arr), eps)
        z = (RHO_real - RHO_null) / sigma

        # Empirical p-value (one-tailed: real >= surrogate)
        p = (np.sum(null_rho_arr >= RHO_real) + 1) / (n_success + 1)

        results.append({
            "patient": pid,
            "seizure": sz,
            "band": band,
            "I": I,
            "sparsity": sparsity,
            "win_idx": win_idx,
            "RHO_real": RHO_real,
            "closed_real": real_closed,
            "closed_null_mean": closed_null_mean,
            "RHO_null": RHO_null,
            "Z": z,
            "p": p,
            "n_channels": n_nodes,
            "n_edges": n_edges,
            "n_triplets_cand": n_triplets,
            "density": density,
            "mean_degree": mean_degree,
            "std_degree": std_degree,
            "swap_efficiency": swap_efficiency,
            "swap_success_rate": swap_success_rate
        })

    return results

# ================= Patient seizure processing (dynamic windows, collect fixed number of valid windows) =================
def process_patient_seizure(data_T, fs, win_L, step, I_list, sparsity_list,
                            pid, sz, band, n_surrogate, base_seed,
                            target_win_N, max_windows):
    """
    data_T : (n_samples, n_channels) already transposed data (time x channels)
    fs : sampling rate
    win_L, step : window length and step size (seconds)
    Returns all valid window results collected (exactly target_win_N for each (I, sp) combination)
    """
    n_samples, n_channels = data_T.shape
    win_len_samples = int(win_L * fs)
    step_samples = int(step * fs)

    # Counters: record how many valid windows have been collected for each (I, sp)
    counters = {(I, sp): 0 for I in I_list for sp in sparsity_list}
    # Store final results
    all_results = []

    win_idx = 0
    start = 0
    while start + win_len_samples <= n_samples and win_idx < max_windows:
        X = data_T[start:start+win_len_samples, :]   # (win_samples, n_channels)

        # Compute results for all I and sparsity values for the current window
        for I in I_list:
            results = process_single_window(X, I, sparsity_list, pid, sz, band, win_idx,
                                            n_surrogate, base_seed)

            for res in results:
                sp = res["sparsity"]
                # Validity check: swap success rate > 0 and p-value is not NaN (surrogate calculation succeeded)
                if res["swap_success_rate"] > 0 and not np.isnan(res["p"]):
                    key = (I, sp)
                    if counters[key] < target_win_N:
                        all_results.append(res)
                        counters[key] += 1

        # Check if all parameter combinations have reached the target count
        if all(count >= target_win_N for count in counters.values()):
            print(f"[INFO] P{pid} SZ{sz} {band} collection complete, stopped early (win_idx={win_idx})")
            break

        start += step_samples
        win_idx += 1

    # Print final collection status for debugging
    print(f"[INFO] P{pid} SZ{sz} {band} final collected windows: {dict(counters)}")
    return all_results

# ================= Main =================
if __name__ == "__main__":
    data_dir = "./data/SEEG"
    save_dir = "./results/data/null_model"
    excel_path = "./data/SEEG/Patient_Information_Table.xlsx"
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_excel(excel_path, sheet_name="ictal")
    df["Pxxx"] = df["Pxxx"].ffill()
    df["patient"] = df["Pxxx"].str.extract(r"P(\d+)").astype(pd.Int64Dtype())
    df["seizure"] = df["SZxx"].str.extract(r"SZ(\d+)").astype(pd.Int64Dtype())
    df = df.dropna(subset=["patient", "seizure"])
    df["patient"] = df["patient"].astype(int)
    df["seizure"] = df["seizure"].astype(int)
    df = df[["patient", "seizure", "Pre", "Episode", "End", "fs"]]
    df = df[df["patient"].isin(target_patients)]
    df = df.sort_values(["patient", "seizure"])

    tasks = []
    for band in bands:
        for _, row in df.iterrows():
            pid = int(row["patient"])
            sz = int(row["seizure"])
            fs = int(row["fs"])

            # HFO band requires sampling rate >= 1000 Hz
            if band == "HFO" and fs < 1000:
                continue

            fname = f"P{pid:03d}_SZ{sz:02d}_{band}.mat"
            fpath = os.path.join(data_dir, fname)
            if not os.path.exists(fpath):
                print(f"[WARNING] File not found: {fname}")
                continue

            try:
                data = load_data_mat(fpath)  # (channels, total_samples)
                Pre = int(row["Pre"])
                Episode = int(row["Episode"])
                End = int(row["End"])

                # Extract ictal segment
                start_idx = (Episode - Pre) * fs
                end_idx = (End - Pre) * fs
                data_ = data[:, start_idx:end_idx]

                # Transpose to (samples, channels)
                data_T = data_.T

                win_L = band_window[band]["win_L"]
                step = band_window[band]["step"]

                # Each patient-seizure-band is a task, internally loops over I_list
                tasks.append((data_T, fs, win_L, step, I_list, sparsity_list,
                              pid, sz, band, n_surrogate, base_random_seed,
                              target_win_N, max_total_windows))

            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    raise
                print(f"[ERROR] P{pid:03d} SZ{sz:02d} {band}: {str(e)}")

    print(f"Total tasks to process: {len(tasks)}")

    # Parallel execution
    results_list = Parallel(n_jobs=N_JOBS, backend="loky", verbose=10)(
        delayed(process_patient_seizure)(*task) for task in tasks
    )

    # Flatten results and save
    all_results = [item for sublist in results_list for item in sublist]
    df_out = pd.DataFrame(all_results)

    # Save in two formats to ensure data safety
    df_out.to_pickle(os.path.join(save_dir, "RHO_Null_Model_All.pkl"))
    df_out.to_csv(os.path.join(save_dir, "RHO_Null_Model_All.csv"), index=False)

    print(f"\n✅ Done! Processed {len(all_results)} results.")
    print(f"Results saved to: {save_dir}")