from scipy.special import binom as binomial
from SPDPE import SPDPE
import itertools
import numpy as np

class simplicial_complex_mvts():
    def __init__(self, multivariate_time_series, I, spar):
        nR, T = np.shape(multivariate_time_series)

        # Variables
        self.raw_data = multivariate_time_series
        self.I = I
        self.sparsity = spar
        self.num_ROI = nR

        # Edges
        self.ets_indexes = {}
        self.ets_max = 0
        self.ets_min = 0

        # Triplets
        self.triplets_indexes = {}
        self.triplets_max = 0
        self.triplets_min = 0

        # Variables for the filtration
        self.list_simplices = []
        self.list_violations = []

        # Initialising the variables by computing the edges and triplets
        self.compute_topological_shape()

    # Initial setup
    def compute_topological_shape(self):
        # -------------------------EDGES-----------------------------
        # Number of edges
        N_edges = int(binomial(self.num_ROI, 2))
        # Indices for the products
        u, v = np.triu_indices(self.num_ROI, k=1, m=self.num_ROI)

        for index in list(zip(u, v)):
            X = self.raw_data[index[0]][:]
            Y = self.raw_data[index[1]][:]
            data = np.vstack((X, Y))
            x = np.transpose(data)
            data_array = SPDPE(x, I=self.I)
            self.ets_min = np.min(np.vstack((self.ets_min, data_array)), axis=0)
            self.ets_max = np.max(np.vstack((self.ets_max, data_array)), axis=0)
        self.ets_indexes = dict(zip(np.arange(N_edges), zip(u, v)))

        # ------------------------TRIPLETS----------------------------
        # Number of triplets
        N_triplets = int(binomial(self.num_ROI, 3))
        # Indices for the products
        self.idx_list_triplets = list(itertools.combinations(range(self.num_ROI), r=3))
        indices = np.array(self.idx_list_triplets)

        for index in indices:
            X = self.raw_data[index[0]][:]
            Y = self.raw_data[index[1]][:]
            Z = self.raw_data[index[2]][:]
            data = np.vstack((X, Y, Z))
            x = np.transpose(data)
            data_array = SPDPE(x, I=self.I)
            self.triplets_min = np.min(np.vstack((self.triplets_min, data_array)), axis=0)
            self.triplets_max = np.max(np.vstack((self.triplets_max, data_array)), axis=0)

        # Saving the indices of all the triplets
        self.triplets_indexes = dict(zip(np.arange(N_triplets), indices))

    def find_max_weight(self):
        edges_abs_max = self.ets_max
        triplets_abs_max = self.triplets_max
        m = np.max([edges_abs_max, triplets_abs_max])
        return (m)

    # Function that creates the list of simplices
    def create_simplicial_complex(self):
        # Creating the list of simplicial complex
        list_simplices = []

        # Selecting the extremal weight between edges and triplets. It will be assigned to all the nodes (i.e. nodes enter at the same instant)
        m_weight = np.min([np.ceil(self.triplets_min), np.ceil(self.ets_min)])

        # Adding all the nodes from the beginning with the same weights
        for i in range(self.num_ROI):
            list_simplices.append(([i], m_weight))

        # -----------Adding the edges:-----------
        for i in self.ets_indexes:
            indexes_ij = self.ets_indexes[i]
            X = self.raw_data[indexes_ij[0]][:]
            Y = self.raw_data[indexes_ij[1]][:]
            data = np.vstack((X, Y))
            x = np.transpose(data)
            weight_current_corrected = SPDPE(x, I=self.I)
            list_simplices.append((indexes_ij, weight_current_corrected))

        # -----------Adding the triplets:-----------
        for i in self.triplets_indexes:
            indexes_ijk = self.triplets_indexes[i]
            X = self.raw_data[indexes_ijk[0]][:]
            Y = self.raw_data[indexes_ijk[1]][:]
            Z = self.raw_data[indexes_ijk[2]][:]
            data = np.vstack((X, Y, Z))
            x = np.transpose(data)
            weight_current_corrected = SPDPE(x, I=self.I)
            list_simplices.append((indexes_ijk, weight_current_corrected))

        list_simplices_for_filtration = self.fix_violations(list_simplices)
        return (list_simplices_for_filtration)

    # Select the top 30% of simplices
    def Sparsity_function(self, data, sparsity = 0.7):
        out = []
        edges_weight = []
        for element in data:
            edges, weight = element
            edges_weight.append(weight)

        value = np.quantile(edges_weight, q = sparsity)

        for element in data:
            edges, weight = element
            if weight >= value:
                out.append((edges, weight))
        return out

    # Function that remove all the violating triangles to create a proper filtration
    def fix_violations(self, list_simplices):
        list_simplices_for_filtration = []

        list_edges_for_permutations = []
        list_triplets_for_permutations = []
        for index, i in enumerate(list_simplices):
            simplices, weight = i
            # # current simplex is a node, included in simplics complexes directly.
            # if len(simplices) == 1:
            #     list_simplices_for_filtration.append((simplices))
            if len(simplices) == 2:
                list_edges_for_permutations.append((simplices, weight))
            if len(simplices) == 3:
                list_triplets_for_permutations.append((simplices, weight))

        list_edges_permutations = self.Sparsity_function(list_edges_for_permutations,sparsity = self.sparsity)
        list_triplets_permutations = self.Sparsity_function(list_triplets_for_permutations, sparsity = self.sparsity)

        set_edges = []
        set_triplets = []

        # current simplex is a edge, included in simplics complexes directly.
        for index, i in enumerate(list_edges_permutations):
            simplices, weight = i
            list_simplices_for_filtration.append((simplices, weight))
            set_edges.append(simplices)

        # current simplex is a triplet, check whether all the sub-simplices have been included.
        for index, i in enumerate(list_triplets_permutations):
            simplices, weight = i
            flag = 0
            for t in itertools.combinations(simplices, 2):
                if t in set_edges:
                    flag += 1
            # If all the sub-simplices already belong to the set, then I add it in the filtration
            if flag == 3:
                list_simplices_for_filtration.append((simplices, weight))
                simplices = list(simplices)
                set_triplets.append(simplices)

        return (list_simplices_for_filtration)

def incidence_matrix_of_simplices(list_simplices, n, m):
    """
    calculate simplicial incidence matrix from list_simplices
    :param list_simplices: a series of simplices
    :param n: the number of nodes
    :param m: the number of simplices
    :return: simplicial incidence matrix
    """
    H = np.zeros((n, m))
    for j in range(m):
        for index in range(len(list_simplices[j])):
            index_x = list_simplices[j][index]
            H[index_x][j] = 1.0
    return H