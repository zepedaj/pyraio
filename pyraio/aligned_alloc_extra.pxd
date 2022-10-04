cimport cython

ctypedef fused T:
    cython.size_t
    cython.longlong

cdef inline T floor(size_t ALIGN_BNDRY, T num_bytes) nogil:
    return ALIGN_BNDRY*<T>(num_bytes/ALIGN_BNDRY)

cdef inline T ceil(size_t ALIGN_BNDRY, T num_bytes) nogil:
    return floor(ALIGN_BNDRY, num_bytes) + ALIGN_BNDRY*<T>(num_bytes%ALIGN_BNDRY>0)
