
cdef extern from "<cstring>" namespace "std":
    void* memcpy( void* dest, void* src, size_t count) nogil
