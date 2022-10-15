# distutils: language = c++

from . cimport liburing
from libcpp.vector cimport vector as cpp_vector
from cython import wraparound, boundscheck
import numpy as np
cimport numpy as np
from libc.errno cimport EAGAIN
from libcpp.string cimport string, to_string

import errno

np.import_array()

cpdef enum:
    PERR_START=1000

cpdef enum PERR:
    PERR_SUBMIT_EINVAL=PERR_START
    PERR_SUBMIT_GET_SQE
    PERR_UNEXPECTED
    PERR_GET_COMPLETED_WRONG_NUM_BYTES

def err_str(val):
    val = abs(val)
    if val < PERR_START:
        return errno.errorcode.get(val, f"<UNKNOWN:{val}>")
    else:
        return str(PERR(val))

cdef class BlockManager:

    cdef liburing.io_uring ring
    cdef size_t depth
    cdef size_t num_submitted
    cdef size_t num_pending
    cdef size_t block_size
    cdef string error_string

    def __cinit__(self, size_t block_size, size_t depth=32):
        cdef int res

        if depth==0 or block_size==0:
            raise Exception(f'Invalid input args (depth={depth}, block_size={block_size}).')

        self.depth = depth
        self.block_size = block_size
        self.num_submitted = 0
        self.num_pending = 0
        self.error_string = b""
        res = liburing.io_uring_queue_init(depth, &self.ring, 0)
        if res != 0:
            raise Exception(f'Error initializing uring: {err_str(res)}.')

    cdef int enqueue(self, int fd, void *buf, unsigned nbytes, liburing.__u64 offset, liburing.__s64 user_data) nogil:
        """
        Adds a pending event to the submission queue. If the queue is full (i.e., if ``num_pending + num_submitted == depth``), a space is ensured
        by submitting all pending and getting at least one completed event.

        :param user_data: **Note**: Although this is a signed value, it must be positive. Error code `PERR_SUBMIT_EINVAL` will be returned otherwise.
        :return: 0 on success; negative PERR or ERRNO on failure.
        """

        cdef liburing.io_uring_sqe *sqe=NULL
        cdef int res
        cdef int attempts = 0

        # Check input
        if user_data <0:
            return -PERR_SUBMIT_EINVAL

        # Ensure an SQE is available.
        if self.num_submitted + self.num_pending == self.depth:
            if not (self.num_submitted>0 and self.get_all_completed(0)>0):
                self.submit()
                self.get_all_completed(1)

        # Submit
        sqe = liburing.io_uring_get_sqe(&self.ring)
        if sqe == NULL:
            return -PERR_SUBMIT_GET_SQE

        # Prepare the submission
        liburing.io_uring_prep_read(sqe, fd, buf, nbytes, offset)
        sqe[0].user_data = user_data
        self.num_pending += 1

        return 0 #Success


    cdef int submit(self) nogil:
        """
        Submits all the events in the submission queue.
        """

        cdef size_t num_submitted

        num_submitted = liburing.io_uring_submit(&self.ring)
        if num_submitted != self.num_pending:
            if num_submitted > 0:
                # Should never happen.
                self.num_submitted += num_submitted
                return -PERR_UNEXPECTED
            else:
                return num_submitted # This is a negated ERRNO.

        self.num_submitted += num_submitted
        self.num_pending -= num_submitted

        return 0


    cdef liburing.__s64 get_completed(self, int blocking=0) nogil:
        """
        Retrieves the next completed event if available, or blocks until available. See also :meth:`get_all_completed`.

        :return: Upon success, returns the retrieved event's ``user_data`` (a positive value). When non-blocking, returns ``-EAGAIN`` if no events are
        available. If blocking and a failure occurs, returns a negated error code.
        """
        cdef liburing.io_uring_cqe *cqe_ptr
        cdef int out

        if blocking:
            out = liburing.io_uring_wait_cqe(&self.ring, &cqe_ptr)
        else:
            out = liburing.io_uring_peek_cqe(&self.ring, &cqe_ptr)
        if out < 0:
            # An error occurred.
            return out
        else:
            self.num_submitted -= 1
            liburing.io_uring_cqe_seen(&self.ring, cqe_ptr)
            if cqe_ptr[0].res < 0:
                # An error occurred.
                return cqe_ptr[0].res
            else:
                # Success
                if <size_t>cqe_ptr[0].res != self.block_size:
                    self.error_string.append(b'Failed to read the expected number of bytes: ').append(to_string(<size_t>cqe_ptr[0].res)).append(b' byte(s) read, but expected ').append(to_string(self.block_size)).append(b'!')
                    return -PERR_GET_COMPLETED_WRONG_NUM_BYTES
                return cqe_ptr[0].user_data


    cdef int get_all_completed(self, int min_num_events=1) nogil:
        """
        Gets all the completed events that are availble, waiting for at least ``min_num_events`` (which can be set to 0). See also :meth:`get_completed`.

        :return: Returns 0 upon success, or negated PERR or ERRNO error code.
        """
        cdef int k
        cdef int num_completed=0

        # Get at least one.
        for k in range(min_num_events):
            out = self.get_completed(True)
            if out<0: # An error occurred
                return out
            num_completed+=1

        # Get all others that are available.
        while True:
            out = self.get_completed()
            if out == -EAGAIN: break
            elif out < 0: # An error occurred
                return out
            num_completed+=1

        return num_completed

    ###################################
    ## For testing purposes
    def _enqueue(self, fd, buf, nbytes, offset, user_data):
        out = self.enqueue(fd, <char *><unsigned long long>buf, nbytes, offset, user_data)
        if out<0:
            raise Exception(f'Error calling `enqueue`: {err_str(out)}')
        return out

    def _submit(self):
        out = self.submit()
        if out<0:
            raise Exception(f'Error calling `submit`: {err_str(out)}')
        return out

    def _get_completed(self, blocking=0):
        out = self.get_completed(blocking)
        if blocking and out<0:
            raise Exception(f'Error calling `get_completed` (blocking): {err_str(out)}')
        elif not blocking and out<0 and out!=EAGAIN:
            #Should never happen.
            raise Exception(f'Error calling `get_completed` (non-blocking): {err_str(out)}')
        else:
            return out
    @property
    def _num_submitted(self):
        return self.num_submitted
    ###################################

    def __dealloc__(self):
         liburing.io_uring_queue_exit(&self.ring)


cdef class RAIOBatchReader:

    cdef BlockManager block_manager
    cdef size_t block_size
    cdef size_t batch_size
    cdef size_t curr_posn
    cdef long long[:] curr_refs
    cdef char[:,:] curr_data

    cdef object ref_map
    cdef np.dtype dtype

    def __cinit__(self, size_t block_size, size_t batch_size, size_t depth=32, object ref_map=None, np.dtype dtype=None):

        self.batch_size = batch_size
        self.block_size = block_size
        self.block_manager = BlockManager(block_size, depth)
        self.curr_posn = batch_size
        self.curr_refs = None
        self.curr_data = None

        self.ref_map = ref_map
        self.dtype = dtype

    cdef int flush(self) nogil:
        if self.block_manager.num_pending>0:
            out = self.block_manager.submit()
            if out<0:
                return out
        out = self.block_manager.get_all_completed(self.block_manager.num_submitted)
        if out<0:
            return out
        return 0


    @boundscheck(False)
    @wraparound(False)
    cdef int enqueue(self, int fd, size_t posn, long long ref) nogil:
        """
        :return: 0 on success, -PERR or -ERRNO on failure.
        """

        cdef int out

        # Add a new batch if necessary
        if self.curr_posn==self.batch_size:
            out = self.flush() # Wait for all pending writes before releasing the memory.
            if out<0:
                return out

            with gil:
                # TODO: How are errors caught here!
                self.curr_refs = np.empty(self.batch_size, dtype=np.longlong)
                self.curr_data = np.empty((self.batch_size, self.block_size), dtype=np.uint8)
                self.curr_posn = 0

        # Enqueue the request
        out = self.block_manager.enqueue(fd, (&(self.curr_data[self.curr_posn,0])), self.block_size, posn, self.curr_posn)
        if out<0:
            return out

        # Update refs
        self.curr_refs[self.curr_posn] = ref
        self.curr_posn+=1

        return out


    def retrieve_batch(self):

        refs = np.asarray(self.curr_refs)
        data = np.asarray(self.curr_data)
        self.curr_refs = None
        self.curr_data = None

        if self.curr_posn>=0:
            refs = refs[:self.curr_posn]
            data = data[:self.curr_posn, :]

        #
        if self.dtype is not None:
            data = data.view(self.dtype)

        #
        if self.ref_map is not None:
            refs = [self.ref_map[k] for k in refs]

        return refs, data

    def iter(self, input_iter, long long default_ref=0):
        """
        """
        cdef int fd
        cdef liburing.__u64 posn
        cdef long long ref=default_ref
        cdef int stop_iter = 0
        cdef int out

        while stop_iter == 0:
            try:
                vals = next(input_iter)
            except StopIteration:
                stop_iter = 1

            if stop_iter == 0:
                if len(vals)==2:
                    fd, posn = vals
                elif len(vals)==3:
                    fd, posn, ref = vals
                else:
                    raise Exception(f'Expected 2 or 3 values but obtained {len(vals)}.')

                #with nogil:
                out = self.enqueue(fd, posn, ref)

                if out<0:
                    self.do_raise(out)

            if self.curr_data is not None and (
                    (stop_iter and self.curr_posn > 0) or
                    self.curr_posn==self.batch_size):
                out = self.flush()
                if out < 0:
                    self.do_raise(out)
                yield self.retrieve_batch()

    def do_raise(self, int err_no):
        if self.block_manager.error_string.size()>0:
            raise Exception(str(self.block_manager.error_string))
        else:
            raise Exception(err_str(err_no))

def raio_batch_read(input_iter, block_size, batch_size, depth=32, ref_map=None, dtype=None):
    rbr = RAIOBatchReader(block_size, batch_size, depth, ref_map=ref_map, dtype=dtype)
    return rbr.iter(input_iter)
