cimport numpy as cnp
from ..criteria.criteria cimport Criteria
cnp.import_array()

cdef class Splitter:
    cdef:
        const double[:, ::1] X
        const double[:, ::1] Y
        int n_features
        int[:] indices
        int n_indices
        Criteria criteria_instance

    cpdef get_split(self, int[::1], int[::1])

cdef class Splitter_DG:
    cdef:
        double[:, ::1] X
        double[:, ::1] Y
        int[::1] E
        int[::1] all_idx
        public double[::1] best_preds
        int n_features
        int[:] indices
        int n_indices
        int[::1] unique_envs

    cpdef get_split(self, int[::1], int[::1], double)
