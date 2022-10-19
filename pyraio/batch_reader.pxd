from libcpp cimport bool as cbool
from libcpp.string cimport string
from . cimport liburing
import numpy as np
cimport numpy as np

np.import_array()

cdef class BaseEventManager:
    cdef string error_string
    cdef int flush(self) nogil
    cdef int enqueue(self, int fd, void *buf, unsigned nbytes, liburing.__u64 offset, cbool skip_ensure_sqe_availability=?) nogil

cdef class RAIOBatchReader:

    cdef BaseEventManager block_manager
    cdef size_t block_size
    cdef size_t batch_size
    cdef size_t curr_posn
    cdef long long[:] curr_refs
    cdef char[:,:] curr_data

    cdef object ref_map
    cdef np.dtype dtype

    cdef int enqueue(self, int fd, size_t posn, long long ref) nogil
