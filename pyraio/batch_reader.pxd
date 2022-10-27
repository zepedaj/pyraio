from libcpp cimport bool as cbool
from libcpp.string cimport string
from . cimport liburing
import numpy as np
cimport numpy as np
from .read_input_iter cimport BaseReadInputIter

np.import_array()

cdef class BaseEventManager:
    cdef string error_string
    cdef int flush(self) nogil
    cdef int enqueue(self, int fd, void *buf, unsigned nbytes, liburing.__u64 offset, cbool skip_ensure_sqe_availability=?) nogil except *

cdef class RAIOBatchReader:

    cdef BaseEventManager event_manager
    cdef size_t block_size
    cdef size_t batch_size
    cdef cbool drop_gil

    cdef object ref_map
    cdef np.dtype dtype

    cdef int build_batch(self, BaseReadInputIter read_input_iter, long long[:] refs, char[:,:] data) nogil except *
