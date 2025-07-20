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

cdef class Splitter_DG_base_v1:
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

cdef class Splitter_DG_base_v2:
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
        object k_to_subtract

    cpdef get_split(self, int[::1], int[::1], double)

cdef class Splitter_DG_fullopt:
    cdef:
        double[:, ::1] X
        double[:, ::1] Y
        int[::1] E
        int[::1] all_idx
        int n_features
        int[:] indices
        int n_indices
        int[::1] unique_envs
        object k_to_subtract

    cpdef get_split(self, int[::1], int[::1], double, list, int)

cdef class Splitter_DG_adafullopt:
    cdef:
        double[:, ::1] X
        double[:, ::1] Y
        int[::1] E
        int[::1] all_idx
        int n_features
        int[:] indices
        int n_indices
        int[::1] unique_envs
        object k_to_subtract

    cpdef get_split(self, int[::1], double, list, list)
