cimport numpy as cnp
from ..criteria.criteria cimport Criteria, Criteria_DG, Criteria_DG_Global
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
        const double[:, ::1] X
        const double[:, ::1] Y
        const int[::1] E
        int n_features
        int[:] indices
        int n_indices
        Criteria_DG criteria_instance

    cpdef get_split(self, int[::1], int[::1], int)

cdef class Splitter_DG_Global:
    cdef:
        const double[:, ::1] X
        const double[:, ::1] Y
        const int[::1] E
        int n_features
        int[:] indices
        int n_indices
        Criteria_DG_Global criteria_instance

    cpdef get_split(self, int[::1], int[::1], int)
