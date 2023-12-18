from libcpp cimport bool as cbool
from libcpp.string cimport string
from . cimport liburing
import numpy as np
cimport numpy as np
from .read_input_iter cimport BaseReadInputIter
from .event_managers cimport BaseEventManager

np.import_array()


cdef class RAIOBatchReader:

    cdef BaseEventManager event_manager
    cdef size_t block_size
    cdef size_t batch_size
    cdef cbool drop_gil

    cdef object ref_map
    cdef np.dtype dtype

    cdef int build_batch(self, BaseReadInputIter read_input_iter, long long[:] refs, char[:,:] data) noexcept nogil
