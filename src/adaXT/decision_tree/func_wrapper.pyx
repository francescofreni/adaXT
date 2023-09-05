cimport numpy as cnp

cdef class FuncWrapper:
    def __cinit__(self):
       self.func = NULL
    
    def crit_func(self, x, y):
        return self.func(x, y)

    @staticmethod
    cdef FuncWrapper make_from_ptr(func_ptr f):
        # TODO do error checking on input function
        cdef FuncWrapper out = FuncWrapper()
        out.func = f
        return out
