cdef extern from "<stdlib.h>" nogil:

    void *aligned_alloc (size_t __alignment, size_t __size)
