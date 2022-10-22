from libcpp.vector cimport vector

cdef struct FilePosn:
    long fd
    long posn
    int key_id

cdef class BaseReadInputIter:
    cdef FilePosn next(self) nogil except *

cdef class ReadInputIterWrapper(BaseReadInputIter):
    cdef object source_iterator
    cdef int default_ref


cdef class ReadInputIterChunk(BaseReadInputIter):
    """
    Returns :class:`BaseReadInputIter`-compatible iterators over chunks of a given read input iterator.
    """
    cdef int posn
    cdef vector[FilePosn] file_posns

    cdef void populate(self, BaseReadInputIter read_input_iter, int chunk_size) nogil except *
