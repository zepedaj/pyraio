# distutils: language = c++

from cpython.ref cimport PyObject

ctypedef struct buf_meta_t:
    size_t data_start # Data start position in aligned buffer
    size_t data_end # Data end position in aligned buffer
    int fd # The original file descriptor
    size_t offset # The original offset requested
    size_t num_bytes # The original num bytes requested
    PyObject *ref # A reference supplied by the user.

cdef str buf_meta_t_str(buf_meta_t &buf_meta)

cdef str syserr_str(int errno)
