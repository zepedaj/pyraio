# distutils: language = c++

import warnings
from cpython.ref cimport PyObject, Py_XINCREF, Py_XDECREF
from cython.view cimport array as cvarray
from . cimport clibaio
from .util cimport buf_meta_t_2, buf_meta_t_2_str, syserr_str
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
import os

from libcpp.set cimport set as cpp_set
# from libcpp.list cimport list as cpp_list
from libcpp.deque cimport deque as cpp_deque
from libcpp.vector cimport vector as cpp_vector
from libcpp cimport bool
from libc.string cimport memcpy

cdef class BlockManager:
    """
    Manages read request submission to clibaio.
    """

    cdef int num_submitted
    cdef size_t depth, block_size

    cdef cpp_vector[clibaio.iocb] _blocks
    """Block memory space"""
    cdef cpp_vector[buf_meta_t_2] _blocks_meta
    """Block meta memory space"""
    cdef cpp_deque[clibaio.iocb *] unused_blocks
    """Pointers to unused blocks"""
    cdef cpp_vector[clibaio.iocb *] pending_blocks
    """Pointers to blocks that are ready to submit to clibaio.io_submit."""
    cdef cpp_vector[clibaio.io_event] completed_events
    """Contains the results of the last call to :meth:`clibaio.io_getevents`."""

    cdef void *aligned_buf_memory
    """Aligned buffer memory."""


    cdef clibaio.io_context_t io_ctx

    cdef inline int append_to_pending(self, int fd, size_t offset, long long ref) nogil except -1
    cdef int submit_pending(self) nogil except -1
    cdef int get_completed(self, long min_nr) nogil except -1
    cdef int release_completed(self) nogil except -1


cdef class RAIOBatchReader:
    cdef size_t batch_size
    cdef list batches
    cdef size_t curr_posn # Where next write must happen in the last batch.
    cdef long long[:] curr_refs
    cdef char[:,:] curr_data
    cdef BlockManager block_manager

    # Used to format the output data
    cdef object ref_map
    cdef np.dtype dtype
    cdef tuple shape # Use -1 for first entry if last batch can have a different size.

    cdef int submit(self, long fd, size_t posn, long long ref) nogil except -1

    cdef int write_completed(self) nogil except -1
