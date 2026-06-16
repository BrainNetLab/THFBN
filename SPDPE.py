import numpy as np

def SPDPE(phase_array, I=20):
    """
    SPDPE function is one of the methods to reproduce the coupling of multiple phases, and the reference is as follows:
    [1]<Measuring multivariate phase synchronization with symbolization and permutation>
    Input:
    phase_array: Phase matrix, where each column represents a node and each row represents a time node
    I: I represents how many segments the phase difference -2*pi~~2*pi is divided into

    Output:
    spdpe: a value between 0 and 1, where the closer the value is to 1, the greater the degree of coupling
    """
    row, col = np.shape(phase_array)

    phase_diff = np.zeros((row, col))  # Allocate space for the phase difference matrix in advance

    for i in np.arange(col):
        column_one = phase_array[:, i: i + 1]  # Take one column of phase values in each cycle

        index = [a for a in np.arange(col) if a != i]  # subscript value

        # In each iteration, the phase matrix is divided into one column and N-1 columns, with this matrix representing the phase values of the N-1 columns
        array_two = phase_array[:, index]

        arraytwo_exp = np.exp(array_two * 1j)  # The entire matrix is multiplied by the imaginary number i and taken to the power of e
        arraytwoexp_sum = np.angle(np.sum(arraytwo_exp, axis=1))  # Add up each row of the matrix and take the phase value

        arraysum_onecolumn = np.reshape(arraytwoexp_sum, (-1, 1))  # Convert the 1-dimensional matrix of the summation result into a 2-dimensional matrix of (-1, 1)
        diff_one = column_one - arraysum_onecolumn
        diff_one = np.reshape(diff_one, (-1, 1))

        phase_diff[:, i: i + 1] = diff_one

    symbolize_index = np.zeros((row, col))  # Symbolized phase difference, allocate storage space in advance

    for ii in np.arange(col):
        column_two = phase_diff[:, [ii]]
        bin_edge = np.histogram_bin_edges(a=column_two, bins=I, range=(-2 * np.pi, 2 * np.pi)) # boundary division
        index_hist = np.searchsorted(a=bin_edge, v=column_two, side='left')
        index_hist = np.reshape(index_hist, (-1, 1))

        symbolize_index[:, [ii]] = index_hist
    _, vector_counts = np.unique(ar=symbolize_index, axis=0, return_counts=True)

    pmf = vector_counts / np.sum(vector_counts)
    logs = np.log2(pmf)
    spdpe = 1 + np.sum(pmf * logs) / np.log2(row+1e-5)

    return spdpe