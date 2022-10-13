# distutils: language = c++

from . cimport liburing
from .util cimport syserr_str
from libcpp.vector cimport vector as cpp_vector

cdef class BlockManager:


    cdef liburing.io_uring ring
    cdef size_t depth
    cdef size_t num_pending

    def __cinit__(self, size_t depth=32):
        cdef int res

        self.depth = depth
        self.num_pending = 0
        res = liburing.io_uring_queue_init(depth, &self.ring, 0)
        if res != 0:
            raise Exception(f'Error initializing uring: {syserr_str(res)}.')

    cdef int submit(self, int fd, void *buf, unsigned nbytes, liburing.__u64 offset, liburing.__u64 user_data) nogil:
        """
        :return: 0 on success; -1 on failure to get SQE from ring; -2 on failure to submit the prepared event.
        """

        cdef liburing.io_uring_sqe *sqe
        cdef int num_submitted
        cdef int res

        sqe = liburing.io_uring_get_sqe(&self.ring)
        if sqe == NULL:
            return -1

        liburing.io_uring_prep_read(sqe, fd, buf, nbytes, offset)
        sqe[0].user_data = user_data

        num_submitted = liburing.io_uring_submit(&self.ring)
        if num_submitted != 1:
            return -2

        self.num_pending+=1

        return 0

    cdef long long get_completed(self, int blocking=0) nogil:
        """
        Retrieves the next completed event is available.

        :return: Upon success, returns the retrieved event's ``user_data`` (a positive value). When non-blocking, returns ``-liburing.EAGAIN`` if no events are
        available. If blocking and a failure occurs, returns a negated error code.
        """
        cdef liburing.io_uring_cqe *cqe_ptr
        cdef long long out

        if blocking:
            out = <int>liburing.io_uring_wait_cqe(&self.ring, &cqe_ptr)
        else:
            out = <int>liburing.io_uring_peek_cqe(&self.ring, &cqe_ptr)
        if out < 0:
            return out
        else:
            self.num_pending -= 1
            liburing.io_uring_cqe_seen(&self.ring, cqe_ptr)
            if cqe_ptr[0].res < 0:
                return cqe_ptr[0].res
            else:
                return <long long>cqe_ptr[0].user_data

    ###################################
    ## For testing purposes
    def _submit(self, fd, buf, nbytes, offset, user_data):
        out = self.submit(fd, <char *><unsigned long long>buf, nbytes, offset, user_data)
        if out<0:
            raise Exception(f'Error calling `submit`: {syserr_str(out)}')

    def _get_completed(self, blocking=0):
        out = self.get_completed(blocking)
        if out<0 and blocking:
            raise Exception(f'Error calling `get_completed`: {syserr_str(out)}')
        else:
            return out
    @property
    def _num_pending(self):
        return self.num_pending
    ###################################


    def __dealloc__(self):
         liburing.io_uring_queue_exit(&self.ring)
