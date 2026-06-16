import numpy as np
import scipy.io as sio  # For reading the matlab .mat format

# Load data in .mat format (rows are ROI, columns are the time instants)
def load_data_mat(path_single_file):
    file_to_open = path_single_file
    data = sio.loadmat(file_to_open)
    key_data = list(data.keys())[-1]
    data = data[key_data]
    return(data)

# Sliding windows (temporal modeling)
def sliding_windows(Z, fs, L_sec, S_sec):
    """
    Z: [n_regions, time]
    return: list of [n_regions, L]
    """
    L = int(L_sec * fs)
    S = int(S_sec * fs)

    T = Z.shape[1]
    starts = np.arange(0, T - L + 1, S)

    windows = [Z[:, s:s + L] for s in starts]
    return windows

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
    data_max = np.nanmax(all_data)

    if data_max == 0:
        return 0, 1, np.linspace(0, 1, max_ticks)

    raw_step = data_max / (max_ticks - 1)

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

    ymax = np.ceil(data_max / step) * step  + step
    ymin = 0

    yticks = np.arange(ymin, ymax + step / 2, step)

    return ymin, ymax, yticks

def cohens_d(x, y):
    x, y = np.array(x), np.array(y)
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std