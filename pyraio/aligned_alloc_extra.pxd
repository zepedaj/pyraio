cimport cython
from cython import cdivision

ctypedef fused T:
    cython.size_t
    cython.longlong

@cdivision(True)
cdef inline T floor(size_t ALIGN_BNDRY, T num_bytes) nogil:
    return ALIGN_BNDRY*<T>(num_bytes/ALIGN_BNDRY)

@cdivision(True)
cdef inline T ceil(size_t ALIGN_BNDRY, T num_bytes) nogil:
    return floor(ALIGN_BNDRY, num_bytes) + ALIGN_BNDRY*<T>(num_bytes%ALIGN_BNDRY>0)
