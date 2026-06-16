from construct_simplicial_complexes import *
from utils import *

def build_incidence_matrix(n_nodes, simplices):
    H = np.zeros((n_nodes, len(simplices)))
    w = np.zeros(len(simplices))
    for j, s in enumerate(simplices):
        nodes, weight = s
        w[j] = weight
        for i in nodes:
            if 0 <= i < n_nodes:
                H[i, j] = 1.0
            else:
                pass
    return H, w

def highorder_state(H, w):
    return H @ w

def build_temporal_highorder_fcn(H_states, K=5, lam=1e-3):
    """
    H_states: list of h(t), each shape (N,)
    return: list of W(k), each shape (N, N)
    """
    Ws = []
    T = len(H_states)

    for k in range(T - K):
        H_win = H_states[k:k + K + 1]

        H_minus = np.stack(H_win[:-1], axis=1)  # N x K
        H_plus  = np.stack(H_win[1:], axis=1)   # N x K

        N = H_minus.shape[0]
        W = H_plus @ H_minus.T @ np.linalg.inv(
            H_minus @ H_minus.T + lam * np.eye(N)
        )
        Ws.append(W)

    return Ws

if __name__ == "__main__":
    filename = "./data/SEEG/P005_SZ01_Gamma.mat"
    data = load_data_mat(filename)

    # parameter
    fs = 1024       # sample rate
    win_L = 1       # window length
    step = 0.25     # step length
    I = 20
    sparsity = 0.6

    Pre = 2787
    Episode = 2793
    End = 3130

    N = data.shape[0]
    win_N = 10   # number of windows
    K = 5

    data_ = data[:, (Episode - Pre) * fs: (End - Pre) * fs]
    windows = sliding_windows(data_, fs, win_L, step)[:win_N]

    H_states = []
    for t, X in enumerate(windows):
            simplicial = simplicial_complex_mvts(X, I, sparsity)
            list_simplices = simplicial.create_simplicial_complex()

            H, w = build_incidence_matrix(N, list_simplices)
            h = highorder_state(H, w)
            H_states.append(h)

    W = build_temporal_highorder_fcn(H_states, K)
    print(W)