from libcpp cimport bool as cbool
from libcpp.string cimport string, to_string
from . cimport liburing
from libcpp.vector cimport vector as cpp_vector
from libcpp.deque cimport deque as cpp_deque

cpdef enum:
    PERR_START=1000

cpdef enum PERR:
    PERR_EINVAL=PERR_START
    PERR_SUBMIT_GET_SQE
    PERR_UNEXPECTED
    PERR_GET_COMPLETED_WRONG_NUM_BYTES

cdef class BaseEventManager:
    cdef string error_string
    cdef int flush(self) noexcept nogil
    cdef int enqueue(self, int fd, void *buf, unsigned nbytes, liburing.__u64 offset, cbool skip_ensure_sqe_availability=?) noexcept nogil

ctypedef struct EventMeta:
    size_t num_bytes
    # Used only by DirectEventManager
    size_t source_num_bytes
    size_t alignment_diff
    void * temp_buf
    void * target_buf

cdef class EventManager(BaseEventManager):
    cdef liburing.io_uring ring
    cdef size_t depth
    cdef size_t num_submitted
    cdef size_t num_pending
    cdef EventMeta *last_enqueued_meta
    cdef EventMeta *last_completed_meta
    cdef cpp_vector[EventMeta] event_metas
    cdef cpp_deque[EventMeta *] available_event_metas

    cdef inline int ensure_sqe_availability(self) noexcept nogil
    cdef int submit(self) noexcept nogil
    cdef int get_completed(self, cbool blocking=?, cbool check_num_bytes=?) noexcept nogil
    cdef int get_all_completed(self, int min_num_events=?) noexcept nogil
    cdef int flush(self) noexcept nogil

cdef class DirectEventManager(EventManager):
    cdef size_t block_size
    cdef size_t alignment
    cdef cpp_vector[char] memory
    cdef cpp_deque[void *] aligned_bufs

    cdef size_t ceil(self, size_t ptr) noexcept nogil
    cdef size_t floor(self, size_t ptr) noexcept nogil
