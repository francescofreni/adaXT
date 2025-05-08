import numpy as np
from collections import Counter
#import cvxpy as cp
from libc.math cimport fabs
from scipy.optimize import root_scalar, root
cimport numpy as cnp
cnp.import_array()
from ..criteria.criteria cimport Criteria
from libc.stdlib cimport qsort

cdef double EPSILON = 2*np.finfo('double').eps
# The rounding error for a criteria function is set twice as large as in DepthTreeBuilder.
# This is needed due to the fact that the criteria does multiple calculations before returing the critical value,
# where the DepthTreeBuilder is just comparing the impurity (that already has gone through this check).

cdef double INFINITY = np.inf

cdef double[:] current_feature_values

cdef inline int compare(const void* a, const void* b) noexcept nogil:
    cdef:
        int a1 = (<int *> a)[0]
        int b1 = (<int *> b)[0]

    if  current_feature_values[a1] >= current_feature_values[b1]:
        return 1
    else:
        return -1

cdef inline int[::1] sort_feature(int[::1] indices):
    """
    Function to sort an array at given indices.

    Parameters
    ----------
    indices : memoryview of NDArray
        A list of the indices which are to be sorted over

    Returns
    -----------
    memoryview of NDArray
        A list of the sorted indices
    """
    cdef:
        int n_obs = indices.shape[0]
        int[::1] ret = indices.copy()
    qsort(&ret[0], n_obs, sizeof(int), compare)
    return ret

# # Old
# cdef double compute_mean_diff(
#            int[:, :] count_L,
#            double[:, :] sum_L,
#            int[:, :] count_R,
#            double[:, :] sum_R,
#            int i,
#        ) nogil:
#            cdef int K = count_L.shape[0]
#            cdef double s = 0.0
#            cdef int c = 0
#            cdef int e
#            cdef double nL, nR, mL, mR
#            for e in range(K):
#                nL = count_L[e, i]
#                nR = count_R[e, i]
#                if nL > 0 and nR > 0:
#                    mL = sum_L[e, i] / nL
#                    mR = sum_R[e, i] / nR
#                    s += fabs(mR - mL)
#                    c += 1
#            return s / c if c > 0 else 0.0

# For the first version
def quad_term(c, S0, S1, n):
    return S0 - 2 * c * S1 + n * c ** 2

# For the first and second version
cdef double compute_max_prop_balance(
    int[:, :] count_L,
    int[:, :] count_R,
    double[:] total_count,
    int i,
) nogil:
    cdef int K = count_L.shape[0]
    cdef double max_left_prop = 0.0
    cdef double max_right_prop = 0.0
    cdef int e
    cdef double left_prop, right_prop, total

    for e in range(K):
        total = total_count[e]
        if total > 0:
            left_prop = count_L[e, i] / total
            right_prop = count_R[e, i] / total

            if left_prop > max_left_prop:
                max_left_prop = left_prop

            if right_prop > max_right_prop:
                max_right_prop = right_prop

    return fabs(max_left_prop - max_right_prop)

# # For the third and fourth version
# cdef double compute_max_prop_balance(
#     int[::1] count_L,
#     int[::1] count_R,
#     double[:] total_count,
# ) nogil:
#     cdef int K = count_L.shape[0]
#     cdef double max_left_prop = 0.0
#     cdef double max_right_prop = 0.0
#     cdef int e
#     cdef double left_prop, right_prop, total
#
#     for e in range(K):
#         total = total_count[e]
#         if total > 0:
#             left_prop = count_L[e] / total
#             right_prop = count_R[e] / total
#
#             if left_prop > max_left_prop:
#                 max_left_prop = left_prop
#
#             if right_prop > max_right_prop:
#                 max_right_prop = right_prop
#
#     return fabs(max_left_prop - max_right_prop)


# For the second version
def best_lam_pairs(lam):
    n_ei, n_ei_L, n_ei_R, mu_ei_L, mu_ei_R, sum_sq_ei, K_ei = env_stats[ei, :]
    n_ej, n_ej_L, n_ej_R, mu_ej_L, mu_ej_R, sum_sq_ej, K_ej = env_stats[ej, :]

    denL = (lam * n_ei_L + (1 - lam) * n_ej_L)
    if denL > 0:
        cL = (lam * n_ei_L * mu_ei_L + (1 - lam) * n_ej_L * mu_ej_L) / denL
    else:
        cL = 0
    denR = (lam * n_ei_R + (1 - lam) * n_ej_R)
    if denR:
        cR = (lam * n_ei_R * mu_ei_R + (1 - lam) * n_ej_R * mu_ej_R) / denR
    else:
        cR = 0

    fi = 1/n_ei * (sum_sq_ei + n_ei_L * cL ** 2 - 2 * n_ei_L * cL * mu_ei_L +
                   n_ei_R * cR ** 2 - 2 * n_ei_R * cR * mu_ei_R + K_ei)

    fj = 1/n_ej * (sum_sq_ej + n_ej_L * cL ** 2 - 2 * n_ej_L * cL * mu_ej_L +
                   n_ej_R * cR ** 2 - 2 * n_ej_R * cR * mu_ej_R + K_ej)

    return fi - fj

def best_lam_triplets(lam):
    n_ei, n_ei_L, n_ei_R, mu_ei_L, mu_ei_R, sum_sq_ei, K_ei = env_stats[ei, :]
    n_ej, n_ej_L, n_ej_R, mu_ej_L, mu_ej_R, sum_sq_ej, K_ej = env_stats[ej, :]
    n_ek, n_ek_L, n_ek_R, mu_ek_L, mu_ek_R, sum_sq_ek, K_ek = env_stats[ek, :]

    li, lj = lam
    lk = 1 - li - lj

    denL = li * n_ei_L + lj * n_ej_L + lk * n_ek_L
    cL = ((li * n_ei_L * mu_ei_L + lj * n_ej_L * mu_ej_L + lk * n_ek_L * mu_ek_L) /
          denL) if denL > 0 else 0
    denR = li * n_ei_R + lj * n_ej_R + lk * n_ek_R
    cR = ((li * n_ei_R * mu_ei_R + lj * n_ej_R * mu_ej_R + lk * n_ek_R * mu_ek_R) /
          denR) if denR > 0 else 0

    fi = 1 / n_ei * (sum_sq_ei + n_ei_L * cL ** 2 - 2 * n_ei_L * cL * mu_ei_L +
                     n_ei_R * cR ** 2 - 2 * n_ei_R * cR * mu_ei_R + K_ei)
    fj = 1 / n_ej * (sum_sq_ej + n_ej_L * cL ** 2 - 2 * n_ej_L * cL * mu_ej_L +
                     n_ej_R * cR ** 2 - 2 * n_ej_R * cR * mu_ej_R + K_ej)
    fk = 1 / n_ek * (sum_sq_ek + n_ek_L * cL ** 2 - 2 * n_ek_L * cL * mu_ek_L +
                     n_ek_R * cR ** 2 - 2 * n_ek_R * cR * mu_ek_R + K_ek)

    return [fi - fj, fj - fk]


cdef class Splitter:
    def __init__(self, double[:, ::1] X, double[:, ::1] Y, criteria_instance: Criteria):
        self.X = X
        self.Y = Y
        self.n_features = X.shape[1]
        self.criteria_instance = criteria_instance

    cpdef get_split(self, int[::1] indices, int[::1] feature_indices):
        global current_feature_values
        self.indices = indices
        self.n_indices = indices.shape[0]
        cdef:
            # number of indices to loop over. Skips last
            int N_i = self.n_indices - 1
            double best_threshold = INFINITY
            double best_score = INFINITY
            int best_feature = 0
            int i, feature  # variables for loop
            int[::1] sorted_index_list_feature
            int[::1] best_sorted
            int best_split_idx
            double crit

        split, best_imp = [], []
        best_split_idx = -1
        best_sorted = None
        # For all features
        for feature in feature_indices:
            current_feature_values = np.asarray(self.X[:, feature])
            sorted_index_list_feature = sort_feature(indices)

            # Loop over sorted feature list
            for i in range(N_i):
                # Skip one iteration of the loop if the current
                # threshold value is the same as the next in the feature list
                if (self.X[sorted_index_list_feature[i], feature] ==
                        self.X[sorted_index_list_feature[i + 1], feature]):
                    continue
                # test the split
                crit, threshold = self.criteria_instance.evaluate_split(
                                                        sorted_index_list_feature, i+1,
                                                        feature
                                                        )
                if best_score > crit:  # rounding error
                    # Save the best split
                    # The index is given as the index of the
                    # first element of the right dataset
                    best_feature, best_threshold = feature, threshold
                    best_score = crit
                    best_split_idx = i + 1
                    best_sorted = sorted_index_list_feature

        # We found a best split
        if best_sorted is not None:
            split = [best_sorted[0:best_split_idx], best_sorted[best_split_idx:self.n_indices]]
            best_imp = [self.criteria_instance.impurity(split[0]), self.criteria_instance.impurity(split[1])]

        return split, best_threshold, best_feature, best_score, best_imp


# First version
#cdef class Splitter_DG:
#
#    def __init__(
#        self,
#        double[:, ::1] X,
#        double[:, ::1] Y,
#        int[::1] E,
#        int[::1] all_idx,
#        double[::1] best_preds
#    ):
#        self.X = X
#        self.Y = Y
#        self.E = E
#        self.n_features = X.shape[1]
#        self.unique_envs = np.ascontiguousarray(np.unique(E))
#        self.best_preds = best_preds
#        self.all_idx = all_idx
#
#    cpdef get_split(
#        self,
#        int[::1] indices,
#        int[::1] feature_indices,
#        double alpha
#    ):
#        global current_feature_values
#        self.indices = indices
#        self.n_indices = indices.shape[0]
#        cdef:
#            int N_i = self.n_indices - 1, K = len(self.unique_envs)
#            double best_threshold = INFINITY, best_score = INFINITY, crit, val
#            int i, feature, split_idx, best_split_idx = -1, idx, best_feature = 0, env, e
#            int[::1] sorted_index_list_feature, best_sorted
#            double[:] sorted_X = np.empty(self.n_indices, dtype=np.double)
#            double[:] sorted_Y = np.empty(self.n_indices, dtype=np.double)
#            int[:] sorted_E = np.empty(self.n_indices, dtype=np.intc)
#            int[:, :] count_mat_L = np.empty((K, self.n_indices + 1), dtype=np.intc)
#            double[:, :] sum_mat_L = np.empty((K, self.n_indices + 1), dtype=np.double)
#            double[:, :] sq_mat_L = np.empty((K, self.n_indices + 1), dtype=np.double)
#            int[:, :] count_mat_R = np.empty((K, self.n_indices + 1), dtype=np.intc)
#            double[:, :] sum_mat_R = np.empty((K, self.n_indices + 1), dtype=np.double)
#            double[:, :] sq_mat_R = np.empty((K, self.n_indices + 1), dtype=np.double)
#            double[::1] total_count = np.zeros(K, dtype=np.double)
#            double[::1] rem_loss_vec = np.zeros(K, dtype=np.double)
#
#        best_sorted = None
#        best_values = np.zeros(2)
#        split = []
#
#        diff = Counter(self.all_idx) - Counter(indices)
#        remaining = np.array(list(diff.elements()), dtype=np.intc)
#
#        for i in range(remaining.shape[0]):
#            idx = remaining[i]
#            env = self.E[idx]
#            val = (self.Y[idx, 0] - self.best_preds[idx]) ** 2
#            rem_loss_vec[env] += val
#
#        print("Remaining: ", len(remaining), np.asarray(rem_loss_vec))
#
#        for i in range(self.all_idx.shape[0]):
#            idx = self.all_idx[i]
#            env = self.E[idx]
#            total_count[env] += 1
#
#        S0_L = cp.Parameter(K, nonneg=True)
#        S1_L = cp.Parameter(K)
#        n_L = cp.Parameter(K, nonneg=True)
#        S0_R = cp.Parameter(K, nonneg=True)
#        S1_R = cp.Parameter(K)
#        n_R = cp.Parameter(K, nonneg=True)
#        R_param = cp.Parameter(K, nonneg=True)
#        #mean_diff_p = cp.Parameter(nonneg=True)
#        max_diff_p = cp.Parameter(nonneg=True)
#
#        cL = cp.Variable(1)
#        cR = cp.Variable(1)
#        t = cp.Variable(nonneg=True)
#
#        constraints = [
#            (quad_term(cL, S0_L, S1_L, n_L)
#             + quad_term(cR, S0_R, S1_R, n_R)
#             + R_param) / total_count <= t
#        ]
#
#        #objective = cp.Minimize(t + alpha * mean_diff_p ** (-1))
#        objective = cp.Minimize(t + alpha * max_diff_p)
#        problem = cp.Problem(objective, constraints)
#
#        for feature in feature_indices:
#            current_feature_values = np.asarray(self.X[:, feature])
#            sorted_index_list_feature = sort_feature(indices)
#
#            for i in range(N_i + 1):
#                idx = sorted_index_list_feature[i]
#                sorted_X[i] = current_feature_values[idx]
#                sorted_Y[i] = self.Y[idx, 0]
#                sorted_E[i] = self.E[idx]
#
#            for e in range(K):
#                count_mat_L[e, 0] = 0
#                sum_mat_L[e, 0] = 0.0
#                sq_mat_L[e, 0] = 0.0
#                for i in range(N_i + 1):
#                    m = 1 if sorted_E[i] == e else 0
#                    count_mat_L[e, i + 1] = count_mat_L[e, i] + m
#                    sum_mat_L[e, i + 1] = sum_mat_L[e, i] + m * sorted_Y[i]
#                    sq_mat_L[e, i + 1] = sq_mat_L[e, i] + m * sorted_Y[i] ** 2
#
#            for e in range(K):
#                count_mat_R[e, N_i + 1] = 0
#                sum_mat_R[e, N_i + 1] = 0.0
#                sq_mat_R[e, N_i + 1] = 0.0
#                for i in range(N_i, -1, -1):
#                    m = 1 if sorted_E[i] == e else 0
#                    count_mat_R[e, i] = count_mat_R[e, i + 1] + m
#                    sum_mat_R[e, i] = sum_mat_R[e, i + 1] + m * sorted_Y[i]
#                    sq_mat_R[e, i] = sq_mat_R[e, i + 1] + m * sorted_Y[i] ** 2
#
#            for i in range(N_i):
#                if (self.X[sorted_index_list_feature[i], feature] ==
#                        self.X[sorted_index_list_feature[i + 1], feature]):
#                    continue
#
#                split_idx = i + 1
#                #mean_diff_p.value = compute_mean_diff(
#                #    count_mat_L, sum_mat_L, count_mat_R, sum_mat_R, split_idx
#                #)
#                max_diff_p.value = compute_max_prop_balance(
#                    count_mat_L, count_mat_R, total_count, split_idx
#                )
#
#                S0_L.value = np.asarray(sq_mat_L[:, split_idx])
#                S1_L.value = np.asarray(sum_mat_L[:, split_idx])
#                n_L.value = np.asarray(count_mat_L[:, split_idx])
#                S0_R.value = np.asarray(sq_mat_R[:, split_idx])
#                S1_R.value = np.asarray(sum_mat_R[:, split_idx])
#                n_R.value = np.asarray(count_mat_R[:, split_idx])
#                R_param.value = np.asarray(rem_loss_vec)
#
#                problem.solve(warm_start=True)
#                crit = problem.value
#
#                if best_score > crit:
#                    best_feature = feature
#                    best_score = crit
#                    best_split_idx = i + 1
#                    best_sorted = sorted_index_list_feature
#                    best_threshold = (
#                        self.X[sorted_index_list_feature[split_idx-1]][feature] +
#                        self.X[sorted_index_list_feature[split_idx]][feature]
#                    ) / 2.0
#                    best_values[0], best_values[1] = cL.value, cR.value
#
#        if best_sorted is not None:
#            split = [best_sorted[0:best_split_idx], best_sorted[best_split_idx:self.n_indices]]
#
#        return split, best_threshold, best_feature, best_score, best_values

# Second version
cdef class Splitter_DG:

    def __init__(
        self,
        double[:, ::1] X,
        double[:, ::1] Y,
        int[::1] E,
        int[::1] all_idx,
        double[::1] best_preds
    ):
        self.X = X
        self.Y = Y
        self.E = E
        self.n_features = X.shape[1]
        self.unique_envs = np.ascontiguousarray(np.unique(E))
        self.best_preds = best_preds
        self.all_idx = all_idx

    cpdef get_split(
        self,
        int[::1] indices,
        int[::1] feature_indices,
        double alpha
    ):
        global current_feature_values
        global env_stats, ei, ej, ek
        self.indices = indices
        self.n_indices = indices.shape[0]
        cdef:
            int N_i = self.n_indices - 1, K = len(self.unique_envs)
            double best_threshold = INFINITY, best_score = INFINITY, crit, val
            int i, feature, split_idx, best_split_idx = -1, idx, best_feature = 0, env, e
            int[::1] sorted_index_list_feature, best_sorted
            double[:] sorted_X = np.empty(self.n_indices, dtype=np.double)
            double[:] sorted_Y = np.empty(self.n_indices, dtype=np.double)
            int[:] sorted_E = np.empty(self.n_indices, dtype=np.intc)
            int[:, :] count_mat_L = np.empty((K, self.n_indices + 1), dtype=np.intc)
            double[:, :] sum_mat_L = np.empty((K, self.n_indices + 1), dtype=np.double)
            double[:, :] sq_mat_L = np.empty((K, self.n_indices + 1), dtype=np.double)
            int[:, :] count_mat_R = np.empty((K, self.n_indices + 1), dtype=np.intc)
            double[:, :] sum_mat_R = np.empty((K, self.n_indices + 1), dtype=np.double)
            double[:, :] sq_mat_R = np.empty((K, self.n_indices + 1), dtype=np.double)
            double[::1] total_count = np.zeros(K, dtype=np.double)
            double[::1] rem_loss_vec = np.zeros(K, dtype=np.double)
            double n_ei, n_ei_L, n_ei_R, mu_ei_L, mu_ei_R, sum_sq_ei, K_ei
            double n_ej, n_ej_L, n_ej_R, mu_ej_L, mu_ej_R, sum_sq_ej, K_ej
            double n_ek, n_ek_L, n_ek_R, mu_ek_L, mu_ek_R, sum_sq_ek, K_ek
            double n_eh, n_eh_L, n_eh_R, mu_eh_L, mu_eh_R, sum_sq_eh, K_eh
            double max_diff, fi, fj, fk, fh, l, fa, fb
            double denL, denR, cL, cR, min_t, best_t_it

        best_sorted = None
        best_values = []
        split = []

        diff = Counter(self.all_idx) - Counter(indices)
        remaining = np.array(list(diff.elements()), dtype=np.intc)

        for i in range(remaining.shape[0]):
            idx = remaining[i]
            env = self.E[idx]
            rem_loss_vec[env] += (self.Y[idx, 0] - self.best_preds[idx]) ** 2

        print("Remaining: ", len(remaining), np.asarray(rem_loss_vec))

        for i in range(self.all_idx.shape[0]):
            idx = self.all_idx[i]
            env = self.E[idx]
            total_count[env] += 1

        for feature in feature_indices:
            current_feature_values = np.asarray(self.X[:, feature])
            sorted_index_list_feature = sort_feature(indices)

            for i in range(N_i + 1):
                idx = sorted_index_list_feature[i]
                sorted_X[i] = current_feature_values[idx]
                sorted_Y[i] = self.Y[idx, 0]
                sorted_E[i] = self.E[idx]

            for e in range(K):
                count_mat_L[e, 0] = 0
                sum_mat_L[e, 0] = 0.0
                sq_mat_L[e, 0] = 0.0
                for i in range(N_i + 1):
                    m = 1 if sorted_E[i] == e else 0
                    count_mat_L[e, i + 1] = count_mat_L[e, i] + m
                    sum_mat_L[e, i + 1] = sum_mat_L[e, i] + m * sorted_Y[i]
                    sq_mat_L[e, i + 1] = sq_mat_L[e, i] + m * sorted_Y[i] ** 2

            for e in range(K):
                count_mat_R[e, N_i + 1] = 0
                sum_mat_R[e, N_i + 1] = 0.0
                sq_mat_R[e, N_i + 1] = 0.0
                for i in range(N_i, -1, -1):
                    m = 1 if sorted_E[i] == e else 0
                    count_mat_R[e, i] = count_mat_R[e, i + 1] + m
                    sum_mat_R[e, i] = sum_mat_R[e, i + 1] + m * sorted_Y[i]
                    sq_mat_R[e, i] = sq_mat_R[e, i + 1] + m * sorted_Y[i] ** 2

            for i in range(N_i):
                if (self.X[sorted_index_list_feature[i], feature] ==
                        self.X[sorted_index_list_feature[i + 1], feature]):
                    continue

                split_idx = i + 1
                max_diff = compute_max_prop_balance(
                    count_mat_L, count_mat_R, total_count, split_idx
                )

                env_stats = np.empty((K, 7), dtype=np.double)
                for e in range(K):
                    env_stats[e, 0] = total_count[e]  # n_e
                    env_stats[e, 1] = count_mat_L[e, i]  # n_e_L
                    env_stats[e, 2] = count_mat_R[e, i]  # n_e_R
                    env_stats[e, 3] = sum_mat_L[e, i] / env_stats[e, 1] if env_stats[e, 1] > 0 else 0.0  # mean_e_L
                    env_stats[e, 4] = sum_mat_R[e, i]  / env_stats[e, 2] if env_stats[e, 2] > 0 else 0.0  # mean_e_R
                    env_stats[e, 5] = sq_mat_L[e, i] + sq_mat_R[e, i]  # sum_sq_e
                    env_stats[e, 6] = rem_loss_vec[e]  # K_e

                best_t_it = INFINITY
                best_vals_it = []

                # Singletons
                min_t = INFINITY
                best_vals_singletons = None
                for ei in range(K):
                    n_ei, n_ei_L, n_ei_R, mu_ei_L, mu_ei_R, sum_sq_ei, K_ei = env_stats[ei, :]
                    cL = mu_ei_L
                    cR = mu_ei_R
                    fi = 1 / n_ei * (sum_sq_ei + n_ei_L * cL ** 2 - 2 * n_ei_L * cL * mu_ei_L +
                                     n_ei_R * cR ** 2 - 2 * n_ei_R * cR * mu_ei_R + K_ei)

                    valid = True
                    for ek in range(K):
                        if ek != ei:
                            n_ek, n_ek_L, n_ek_R, mu_ek_L, mu_ek_R, sum_sq_ek, K_ek = env_stats[ek, :]
                            fk = 1 / n_ek * (sum_sq_ek + n_ek_L * cL ** 2 - 2 * n_ek_L * cL * mu_ek_L +
                                             n_ek_R * cR ** 2 - 2 * n_ek_R * cR * mu_ek_R + K_ek)
                            if fk > fi:
                                valid = False
                                break

                    if valid and fi < min_t:
                        min_t = fi
                        best_vals_singletons = [cL, cR]

                if min_t < best_t_it:
                    best_t_it = min_t
                    best_vals_it = best_vals_singletons

                # Pairs
                if K >= 2:
                    min_t = INFINITY
                    best_vals_pairs = None
                    for ei in range(K):
                        n_ei, n_ei_L, n_ei_R, mu_ei_L, mu_ei_R, sum_sq_ei, K_ei = env_stats[ei, :]
                        for ej in range(ei + 1, K):
                            n_ej, n_ej_L, n_ej_R, mu_ej_L, mu_ej_R, sum_sq_ej, K_ej = env_stats[ej, :]

                            fa = best_lam_pairs(0)
                            fb = best_lam_pairs(1)
                            if fa * fb < 0 or abs(fa) < 1e-8 or abs(fb) < 1e-8:
                                if abs(fa) < 1e-8:
                                    l = 0.0
                                elif abs(fb) < 1e-8:
                                    l = 1.0
                                else:
                                    l = root_scalar(best_lam_pairs, bracket=[0, 1], method='brentq').root

                                denL = l * n_ei_L + (1 - l) * n_ej_L
                                cL = (l * n_ei_L * mu_ei_L + (1 - l) * n_ej_L * mu_ej_L) / denL if denL > 0 else 0

                                denR = l * n_ei_R + (1 - l) * n_ej_R
                                cR = (l * n_ei_R * mu_ei_R + (1 - l) * n_ej_R * mu_ej_R) / denR if denR > 0 else 0

                                fi = 1 / n_ei * (sum_sq_ei + n_ei_L * cL ** 2 - 2 * n_ei_L * cL * mu_ei_L +
                                                 n_ei_R * cR ** 2 - 2 * n_ei_R * cR * mu_ei_R + K_ei)

                                valid = True
                                for ek in range(K):
                                    if ek != ei and ek != ej:
                                        n_ek, n_ek_L, n_ek_R, mu_ek_L, mu_ek_R, sum_sq_ek, K_ek = env_stats[ek, :]
                                        fk = 1 / n_ek * (sum_sq_ek + n_ek_L * cL ** 2 - 2 * n_ek_L * cL * mu_ek_L +
                                                         n_ek_R * cR ** 2 - 2 * n_ek_R * cR * mu_ek_R + K_ek)
                                        if fk > fi:
                                            valid = False
                                            break

                                if valid and fi < min_t:
                                    min_t = fi
                                    best_vals_pairs = [cL, cR]

                    if min_t < best_t_it:
                        best_t_it = min_t
                        best_vals_it = best_vals_pairs

                # Triplets
                if K >= 3:
                    min_t = INFINITY
                    best_vals_triplets = None
                    for ei in range(K):
                        n_ei, n_ei_L, n_ei_R, mu_ei_L, mu_ei_R, sum_sq_ei, K_ei = env_stats[ei, :]
                        for ej in range(ei + 1, K):
                            n_ej, n_ej_L, n_ej_R, mu_ej_L, mu_ej_R, sum_sq_ej, K_ej = env_stats[ej, :]
                            for ek in range(ej + 1, K):
                                n_ek, n_ek_L, n_ek_R, mu_ek_L, mu_ek_R, sum_sq_ek, K_ek = env_stats[ek, :]

                                init = np.array([1 / 3, 1 / 3])

                                sol = root(best_lam_triplets, init, method='lm')
                                if sol.success:
                                    li, lj = sol.x
                                    lk = 1 - li - lj

                                    denL = li * n_ei_L + lj * n_ej_L + lk * n_ek_L
                                    cL = ((li * n_ei_L * mu_ei_L + lj * n_ej_L * mu_ej_L + lk * n_ek_L * mu_ek_L) /
                                          denL) if denL > 0 else 0

                                    denR = li * n_ei_R + lj * n_ej_R + lk * n_ek_R
                                    cR = ((li * n_ei_R * mu_ei_R + lj * n_ej_R * mu_ej_R + lk * n_ek_R * mu_ek_R) /
                                          denR) if denR > 0 else 0

                                    fi = 1 / n_ei * (sum_sq_ei + n_ei_L * cL ** 2 - 2 * n_ei_L * cL * mu_ei_L +
                                                     n_ei_R * cR ** 2 - 2 * n_ei_R * cR * mu_ei_R + K_ei)
                                    fj = 1 / n_ej * (sum_sq_ej + n_ej_L * cL ** 2 - 2 * n_ej_L * cL * mu_ej_L +
                                                     n_ej_R * cR ** 2 - 2 * n_ej_R * cR * mu_ej_R + K_ej)
                                    fk = 1 / n_ek * (sum_sq_ek + n_ek_L * cL ** 2 - 2 * n_ek_L * cL * mu_ek_L +
                                                     n_ek_R * cR ** 2 - 2 * n_ek_R * cR * mu_ek_R + K_ek)

                                    f_values = [fi, fj, fk]
                                    max_f = fi
                                    for h in range(3):
                                        if f_values[h] > max_f:
                                            max_f = f_values[h]

                                    valid = True
                                    for eh in range(K):
                                        if eh != ei and eh != ej and eh != ek:
                                            n_eh, n_eh_L, n_eh_R, mu_eh_L, mu_eh_R, sum_sq_eh, K_eh = env_stats[eh, :]
                                            fh = 1 / n_eh * (sum_sq_eh + n_eh_L * cL ** 2 - 2 * n_eh_L * cL * mu_eh_L +
                                                             n_eh_R * cR ** 2 - 2 * n_eh_R * cR * mu_eh_R + K_eh)
                                            if fh > max_f:
                                                valid = False
                                                break

                                    if valid and max_f < min_t:
                                        min_t = max_f
                                        best_vals_triplets = [cL, cR]

                    if min_t < best_t_it:
                        best_t_it = min_t
                        best_vals_it = best_vals_triplets

                crit = best_t_it + alpha * max_diff

                if best_score > crit:
                    best_feature = feature
                    best_score = crit
                    best_split_idx = i + 1
                    best_sorted = sorted_index_list_feature
                    best_threshold = (
                        self.X[sorted_index_list_feature[split_idx-1]][feature] +
                        self.X[sorted_index_list_feature[split_idx]][feature]
                    ) / 2.0
                    best_values = best_vals_it

        if best_sorted is not None:
            split = [best_sorted[0:best_split_idx], best_sorted[best_split_idx:self.n_indices]]

        return split, best_threshold, best_feature, best_score, best_values

# # Third version
# cdef class Splitter_DG:
#
#     def __init__(
#        self,
#        double[:, ::1] X,
#        double[:, ::1] Y,
#        int[::1] E,
#        int[::1] all_idx,
#     ):
#         self.X = X
#         self.Y = Y
#         self.E = E
#         self.n_features = X.shape[1]
#         self.unique_envs = np.ascontiguousarray(np.unique(E))
#         self.all_idx = all_idx
#
#     cpdef get_split(
#         self,
#         int[::1] indices,
#         int[::1] feature_indices,
#         double alpha,
#         list nodes_indices,
#         int pos,
#     ):
#         global current_feature_values
#         self.indices = indices
#         self.n_indices = indices.shape[0]
#         cdef:
#             int N_i = self.n_indices - 1, K = len(self.unique_envs)
#             double best_threshold = INFINITY, best_score = INFINITY, crit, val
#             int i, feature, split_idx, best_split_idx = -1, idx, best_feature = 0, env, e
#             int[::1] sorted_index_list_feature, best_sorted
#             int n_nodes = len(nodes_indices)
#             int n_values = n_nodes + 1
#             double[::1] total_count = np.zeros(K, dtype=np.double)
#             int[::1] count_L = np.empty(K, dtype=np.intc), count_R = np.empty(K, dtype=np.intc)
#             double[:, :] S0_mat = np.zeros((K, n_values), dtype=np.double)
#             double[:, :] S1_mat = np.zeros((K, n_values), dtype=np.double)
#             int[:, :] n_mat = np.zeros((K, n_values), dtype=np.intc)
#
#         E_np = np.asarray(self.E)
#         Y_np = np.asarray(self.Y)
#
#         best_sorted = None
#         best_values = np.zeros(n_values)
#         split = []
#
#         for i in range(self.all_idx.shape[0]):
#             idx = self.all_idx[i]
#             env = self.E[idx]
#             total_count[env] += 1
#
#         S0 = cp.Parameter((K, n_values), nonneg=True)
#         S1 = cp.Parameter((K, n_values))
#         n = cp.Parameter((K, n_values), nonneg=True)
#         max_diff_p = cp.Parameter(nonneg=True)
#
#         c_values = cp.Variable(n_values)
#         t = cp.Variable(nonneg=True)
#
#         constraints = []
#
#         for e in range(K):
#             expr = 0
#             for i in range(n_nodes):
#                 if i != pos:
#                     idxs = nodes_indices[i]
#                     j = i
#                     if j > pos:
#                         j += 1
#                     Y_node = Y_np[idxs, 0]
#                     E_node = E_np[idxs]
#                     mask = E_node == e
#                     S0_mat[e, j] = np.sum(Y_node[mask] ** 2)
#                     S1_mat[e, j] = np.sum(Y_node[mask])
#                     n_mat[e, j] = np.sum(mask)
#             for i in range(n_values):
#                 expr += S0[e, i] - 2 * c_values[i] * S1[e, i] + n[e, i] * c_values[i] ** 2
#             constraints.append(expr / total_count[e] <= t)
#
#         objective = cp.Minimize(t + alpha * max_diff_p)
#         problem = cp.Problem(objective, constraints)
#
#         for feature in feature_indices:
#             current_feature_values = np.asarray(self.X[:, feature])
#             sorted_index_list_feature = sort_feature(indices)
#
#             for i in range(N_i):
#                 if (self.X[sorted_index_list_feature[i], feature] ==
#                         self.X[sorted_index_list_feature[i + 1], feature]):
#                     continue
#
#                 split_idx = i + 1
#
#                 for e in range(K):
#                     count_L[e] = np.sum(E_np[sorted_index_list_feature][:split_idx] == e)
#                     count_R[e] = np.sum(E_np[sorted_index_list_feature][split_idx:] == e)
#
#                 max_diff_p.value = compute_max_prop_balance(
#                     count_L, count_R, total_count,
#                 )
#
#                 for e in range(K):
#                     idxs = sorted_index_list_feature[:split_idx]
#                     Y_node = Y_np[idxs, 0]
#                     E_node = E_np[idxs]
#                     mask = E_node == e
#                     S0_mat[e, pos] = np.sum(Y_node[mask] ** 2)
#                     S1_mat[e, pos] = np.sum(Y_node[mask])
#                     n_mat[e, pos] = np.sum(mask)
#
#                     idxs = sorted_index_list_feature[split_idx:]
#                     Y_node = Y_np[idxs, 0]
#                     E_node = E_np[idxs]
#                     mask = E_node == e
#                     S0_mat[e, pos+1] = np.sum(Y_node[mask] ** 2)
#                     S1_mat[e, pos+1] = np.sum(Y_node[mask])
#                     n_mat[e, pos+1] = np.sum(mask)
#
#                 S0.value = np.asarray(S0_mat)
#                 S1.value = np.asarray(S1_mat)
#                 n.value = np.asarray(n_mat)
#
#                 problem.solve(warm_start=True)
#                 crit = problem.value
#
#                 if best_score > crit:
#                     best_feature = feature
#                     best_score = crit
#                     best_split_idx = i + 1
#                     best_sorted = sorted_index_list_feature
#                     best_threshold = (
#                         self.X[sorted_index_list_feature[split_idx-1]][feature] +
#                         self.X[sorted_index_list_feature[split_idx]][feature]
#                     ) / 2.0
#                     best_values = c_values.value
#
#         if best_sorted is not None:
#             split = [best_sorted[0:best_split_idx], best_sorted[best_split_idx:self.n_indices]]
#
#         return split, best_threshold, best_feature, best_score, best_values

# # Fourth version
# cdef class Splitter_DG:
#
#     def __init__(
#        self,
#        double[:, ::1] X,
#        double[:, ::1] Y,
#        int[::1] E,
#        int[::1] all_idx,
#     ):
#         self.X = X
#         self.Y = Y
#         self.E = E
#         self.n_features = X.shape[1]
#         self.unique_envs = np.ascontiguousarray(np.unique(E))
#         self.all_idx = all_idx
#
#     cpdef get_split(
#         self,
#         int[::1] feature_indices,
#         double alpha,
#         list nodes_indices,
#         list mask_nodes,
#     ):
#         global current_feature_values
#         cdef:
#             int K = len(self.unique_envs)
#             double best_threshold = INFINITY, best_score = INFINITY, crit, val
#             int i, feature, split_idx, best_split_idx = -1, idx, best_feature = 0, env, e, best_node_idx = 0
#             int[::1] sorted_index_list_feature, best_sorted
#             int n_nodes = len(nodes_indices)
#             int n_values = n_nodes + 1
#             double[::1] total_count = np.zeros(K, dtype=np.double)
#             int[::1] count_L = np.empty(K, dtype=np.intc), count_R = np.empty(K, dtype=np.intc)
#             double[:, :] S0_mat = np.zeros((K, n_values), dtype=np.double)
#             double[:, :] S1_mat = np.zeros((K, n_values), dtype=np.double)
#             int[:, :] n_mat = np.zeros((K, n_values), dtype=np.intc)
#
#         E_np = np.asarray(self.E)
#         Y_np = np.asarray(self.Y)
#
#         best_sorted = None
#         best_values = np.zeros(n_values)
#         split = []
#
#         for i in range(self.all_idx.shape[0]):
#             idx = self.all_idx[i]
#             env = self.E[idx]
#             total_count[env] += 1
#
#         S0 = cp.Parameter((K, n_values), nonneg=True)
#         S1 = cp.Parameter((K, n_values))
#         n = cp.Parameter((K, n_values), nonneg=True)
#         max_diff_p = cp.Parameter(nonneg=True)
#
#         c_values = cp.Variable(n_values)
#         t = cp.Variable(nonneg=True)
#
#         constraints = []
#
#         for e in range(K):
#             expr = 0
#             for i in range(n_values):
#                 expr += S0[e, i] - 2 * c_values[i] * S1[e, i] + n[e, i] * c_values[i] ** 2
#             constraints.append(expr / total_count[e] <= t)
#
#         objective = cp.Minimize(t + alpha * max_diff_p)
#         problem = cp.Problem(objective, constraints)
#
#         for pos, indices in enumerate(nodes_indices):
#             if mask_nodes[pos]:
#                 updated = False
#                 for feature in feature_indices:
#                     current_feature_values = np.asarray(self.X[:, feature])
#                     sorted_index_list_feature = sort_feature(indices)
#
#                     N_i = indices.shape[0] - 1
#
#                     for i in range(N_i):
#                         if (self.X[sorted_index_list_feature[i], feature] ==
#                                 self.X[sorted_index_list_feature[i + 1], feature]):
#                             continue
#
#                         split_idx = i + 1
#
#                         for e in range(K):
#                             count_L[e] = np.sum(E_np[sorted_index_list_feature][:split_idx] == e)
#                             count_R[e] = np.sum(E_np[sorted_index_list_feature][split_idx:] == e)
#
#                         max_diff_p.value = compute_max_prop_balance(
#                             count_L, count_R, total_count,
#                         )
#
#                         for e in range(K):
#                             for j in range(n_nodes):
#                                 if j != pos:
#                                     idxs = nodes_indices[j]
#                                     h = j
#                                     if h > pos:
#                                         h += 1
#                                     Y_node = Y_np[idxs, 0]
#                                     E_node = E_np[idxs]
#                                     mask = E_node == e
#                                     S0_mat[e, h] = np.sum(Y_node[mask] ** 2)
#                                     S1_mat[e, h] = np.sum(Y_node[mask])
#                                     n_mat[e, h] = np.sum(mask)
#
#                             idxs = sorted_index_list_feature[:split_idx]
#                             Y_node = Y_np[idxs, 0]
#                             E_node = E_np[idxs]
#                             mask = E_node == e
#                             S0_mat[e, pos] = np.sum(Y_node[mask] ** 2)
#                             S1_mat[e, pos] = np.sum(Y_node[mask])
#                             n_mat[e, pos] = np.sum(mask)
#
#                             idxs = sorted_index_list_feature[split_idx:]
#                             Y_node = Y_np[idxs, 0]
#                             E_node = E_np[idxs]
#                             mask = E_node == e
#                             S0_mat[e, pos + 1] = np.sum(Y_node[mask] ** 2)
#                             S1_mat[e, pos + 1] = np.sum(Y_node[mask])
#                             n_mat[e, pos + 1] = np.sum(mask)
#
#                         S0.value = np.asarray(S0_mat)
#                         S1.value = np.asarray(S1_mat)
#                         n.value = np.asarray(n_mat)
#
#                         problem.solve(warm_start=True)
#                         crit = problem.value
#
#                         if best_score > crit:
#                             updated = True
#                             best_feature = feature
#                             best_score = crit
#                             best_split_idx = i + 1
#                             best_sorted = sorted_index_list_feature
#                             best_threshold = (
#                                 self.X[sorted_index_list_feature[split_idx - 1]][feature] +
#                                 self.X[sorted_index_list_feature[split_idx]][feature]
#                             ) / 2.0
#                             best_values = c_values.value
#                 if updated:
#                     best_node_idx = pos
#
#         if best_sorted is not None:
#             split = [best_sorted[0:best_split_idx], best_sorted[best_split_idx:]]
#
#         return split, best_threshold, best_feature, best_score, best_values, best_node_idx
