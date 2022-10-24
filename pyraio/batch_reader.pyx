# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8

from typing import Union, Iterable, Tuple
from . cimport liburing
from .read_input_iter cimport FilePosn, ReadInputIterWrapper, BaseReadInputIter
from libcpp.vector cimport vector as cpp_vector
from libcpp.deque cimport deque as cpp_deque
from cython cimport wraparound, boundscheck, final, cdivision, binding
import numpy as np
cimport numpy as np
from libc.errno cimport EAGAIN
from libcpp.string cimport string, to_string
from libcpp cimport bool as cbool
from .util cimport memcpy

import errno

np.import_array()

cdef int DEFAULT_DEPTH = 32
cdef cbool DEFAULT_BLOCKING=False
cdef cbool DEFAULT_CHECK_NUM_BYTES=True
cdef cbool DEFAULT_SKIP_ENSURE_SQE_AVAILABILITY=False

cpdef enum:
    PERR_START=1000

cpdef enum PERR:
    PERR_EINVAL=PERR_START
    PERR_SUBMIT_GET_SQE
    PERR_UNEXPECTED
    PERR_GET_COMPLETED_WRONG_NUM_BYTES

def err_str(val):
    val = abs(val)
    if val < PERR_START:
        return errno.errorcode.get(val, f"<UNKNOWN:{val}>")
    else:
        return str(PERR(val))

ctypedef struct EventMeta:
    size_t num_bytes
    # Used only by DirectEventManager
    size_t source_num_bytes
    size_t alignment_diff
    void * temp_buf
    void * target_buf

cdef class BaseEventManager:
    cdef int flush(self) nogil:
        raise NotImplementedError('Abstract class.')
    cdef int enqueue(self, int fd, void *buf, unsigned nbytes, liburing.__u64 offset, cbool skip_ensure_sqe_availability=DEFAULT_SKIP_ENSURE_SQE_AVAILABILITY) nogil except *:
        raise NotImplementedError('Abstract class.')

cdef class EventManager(BaseEventManager):

    cdef liburing.io_uring ring
    cdef size_t depth
    cdef size_t num_submitted
    cdef size_t num_pending
    cdef EventMeta *last_enqueued_meta
    cdef EventMeta *last_completed_meta
    cdef cpp_vector[EventMeta] event_metas
    cdef cpp_deque[EventMeta *] available_event_metas

    def __init__(self, size_t depth=DEFAULT_DEPTH):
        cdef int res

        if depth==0:
            raise Exception(f'Invalid input args (depth={depth}).')

        self.depth = depth
        self.num_submitted = 0
        self.num_pending = 0
        self.error_string = b""

        # Initialize the meta pool
        self.event_metas.resize(depth)
        for k in range(depth):
            self.available_event_metas.push_front(&(self.event_metas[k]))

        # Initialize the uring
        res = liburing.io_uring_queue_init(depth, &self.ring, 0)
        if res != 0:
            raise Exception(f'Error initializing uring: {err_str(res)}.')

    cdef inline int ensure_sqe_availability(self) nogil:
        cdef int out = 0
        # Ensure an SQE is available
        if self.num_submitted + self.num_pending == self.depth:
            out = self.submit()
            if out<0:
                return out
            out = self.get_all_completed(1)

        return out

    cdef int enqueue(self, int fd, void *buf, unsigned nbytes, liburing.__u64 offset, cbool skip_ensure_sqe_availability=DEFAULT_SKIP_ENSURE_SQE_AVAILABILITY) nogil:
        """
        Adds a pending event to the submission queue. If the queue is full (i.e., if ``num_pending + num_submitted == depth``), a space is ensured
        by submitting all pending and getting at least one completed event.

        :return: 0 on success; negative PERR or ERRNO on failure.
        """

        cdef liburing.io_uring_sqe *sqe=NULL
        cdef int res
        cdef int attempts = 0

        # Reserve it
        if not skip_ensure_sqe_availability:
            out = self.ensure_sqe_availability()
            if out < 0: return out
        sqe = liburing.io_uring_get_sqe(&self.ring)
        if sqe == NULL:
            return -PERR_SUBMIT_GET_SQE

        # Prepare the submission
        liburing.io_uring_prep_read(sqe, fd, buf, nbytes, offset)
        self.num_pending += 1

        # Prepare event meta
        if self.available_event_metas.size()==0:
            self.error_string.append(b'Available event metas unexpectedly empty!')
            return -PERR.PERR_UNEXPECTED
        meta = self.available_event_metas.front()
        self.available_event_metas.pop_front()
        meta[0].num_bytes = nbytes

        self.last_enqueued_meta = meta
        sqe[0].user_data = <liburing.__u64>meta

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
                self.error_string.append(b'Expected ').append(to_string(self.num_pending)).append(b' events submitted, but submitted ').append(to_string(num_submitted)).append(b'.')
                return -PERR_UNEXPECTED
            else:
                return num_submitted # This is a negated ERRNO.

        self.num_submitted += num_submitted
        self.num_pending -= num_submitted

        return 0


    cdef int get_completed(self, cbool blocking=DEFAULT_BLOCKING, cbool check_num_bytes=DEFAULT_CHECK_NUM_BYTES) nogil:
        """
        Retrieves the next completed event if available, or blocks until available. See also :meth:`get_all_completed`.

        :param blocking: Whether to wait until at least one event is available.
        :param check_num_bytes: Whether to check whether the expected number of bytes was returned.

        :return: Upon success, returns 0. When non-blocking, returns ``-EAGAIN`` if no events are
        available. If blocking and a failure occurs, returns a negated error code. If an event is retrieved, :attr:`last_completed_meta` is set to point to that event's meta data and the
        number of read bytes is returned.
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
                meta = <EventMeta *>(cqe_ptr[0].user_data)
                if check_num_bytes and <size_t>cqe_ptr[0].res != meta.num_bytes:
                    self.error_string.append(
                        b'Failed to read the expected number of bytes: ').append(
                            to_string(<size_t>cqe_ptr[0].res)).append(
                                b' byte(s) read, but expected ').append(
                                    to_string(meta.num_bytes)).append(b'!')
                    return -PERR_GET_COMPLETED_WRONG_NUM_BYTES
                self.last_completed_meta = <EventMeta *>cqe_ptr[0].user_data
                self.available_event_metas.push_front(meta)
                return cqe_ptr[0].res


    cdef int get_all_completed(self, int min_num_events=1) nogil:
        """
        Gets all the completed events that are availble, waiting for at least ``min_num_events`` (which can be set to 0). See also :meth:`get_completed`.

        :return: Returns the number of completed events, or a negated PERR or ERRNO error code.
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

    cdef int flush(self) nogil:
        """ Submits all pending events, if any, and waits for the completion of all submitted events."""

        if self.num_pending>0:
            out = self.submit()
            if out<0:
                return out
        out = self.get_all_completed(self.num_submitted)
        if out<0:
            return out
        return 0

    ###################################
    ## For testing purposes
    def _enqueue(self, fd, buf, nbytes, offset):
        out = self.enqueue(fd, <char *><unsigned long long>buf, nbytes, offset)
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


cdef class DirectEventManager(EventManager):
    """
    Transparently aligns read offsets, num_bytes, and block sizes in order to support files opened in O_DIRECT mode.
    """
    cdef size_t block_size
    cdef size_t alignment
    cdef cpp_vector[char] memory
    cdef cpp_deque[void *] aligned_bufs

    def __init__(self, size_t max_block_size, size_t alignment=512, **kwargs):
        """
        Use this when using file descriptors opened with the O_DIRECT option.
        """
        cdef size_t k, aligned_block_size

        if max_block_size==0 or alignment==0:
            raise Exception(f'Invalid input args (max_block_size={max_block_size}, alignment={alignment}).')
        self.block_size = max_block_size
        self.alignment = alignment
        super().__init__(**kwargs)

        aligned_block_size = self.ceil(max_block_size + alignment - 1)
        self.memory.resize(self.ceil(self.depth*aligned_block_size + alignment - 1))

        for k in range(self.depth):
            self.aligned_bufs.push_back(<void *>(self.ceil(<size_t>self.memory.data()) + k*aligned_block_size))

    @cdivision(True)
    cdef size_t ceil(self, size_t ptr) nogil:
        return self.alignment * (ptr//self.alignment + (ptr%self.alignment>0))

    @cdivision(True)
    cdef size_t floor(self, size_t ptr) nogil:
        return self.alignment * (ptr//self.alignment)

    cdef int enqueue(self, int fd, void *buf, unsigned nbytes, liburing.__u64 offset, cbool skip_ensure_sqe_availability=DEFAULT_SKIP_ENSURE_SQE_AVAILABILITY) nogil:

        cdef size_t offset_floor, alignment_diff


        if nbytes>self.block_size:
            self.error_string.append(b'Invalid value nbytes=').append(to_string(nbytes)).append(b' > self.block_size=').append(to_string(self.block_size))
            return -PERR_EINVAL

        # Get an aligned buf
        out = self.ensure_sqe_availability() # Also ensures aligned buf availability.
        if out < 0: return out
        if self.aligned_bufs.size() == 0:
            self.error_string.append(b'Aligned bufs unexpectedly empty!')
            return -PERR_UNEXPECTED
        aligned_buf = self.aligned_bufs.front()
        self.aligned_bufs.pop_front()

        offset_floor = self.floor(offset)
        alignment_diff = offset - offset_floor
        out = EventManager.enqueue(self, fd, aligned_buf, self.ceil(nbytes + alignment_diff), offset_floor, True)
        if out<0:
            return out
        self.last_enqueued_meta[0].temp_buf = <void *>(<size_t>aligned_buf + alignment_diff)
        self.last_enqueued_meta[0].target_buf = buf
        self.last_enqueued_meta[0].source_num_bytes = nbytes
        self.last_enqueued_meta[0].alignment_diff = alignment_diff

        return out

    cdef int get_completed(self, cbool blocking=DEFAULT_BLOCKING, cbool check_num_bytes=DEFAULT_CHECK_NUM_BYTES) nogil:
        out = EventManager.get_completed(self, blocking, False)
        if out<0:
            return out
        meta = self.last_completed_meta[0]
        min_num_bytes = meta.source_num_bytes + meta.alignment_diff
        if check_num_bytes and <size_t>out<min_num_bytes:
            self.error_string.append(
                b'Failed to read the expected number of bytes: ').append(
                    to_string(out)).append(
                        b' byte(s) read, but expected at least ').append(
                            to_string(min_num_bytes)).append(b'!')
            return -PERR_GET_COMPLETED_WRONG_NUM_BYTES
        memcpy(meta.target_buf, meta.temp_buf, meta.source_num_bytes)
        self.aligned_bufs.push_front(<void *>self.floor(<size_t>meta.temp_buf))


DEFAULT_DIRECT = False

cdef class RAIOBatchReader:
    """
    Iterates over batches read from a binary file. Reads can occur in random order. Each output batch contains blocks of size math:`N \times S`, where :math:`N` is the batch size and :math:`S` is the block size.
    The returned batch is in numpy array form. By default, the dtype is ``uint8``, but a dtype can be specified as well.

    :param block_size: The size :math:`S` of each block.
    :param batch_size: The number of blocks :math:`N` in each batch.
    :param depth: The maximum number of low-level read requests occuring simultaneously.
    :param ref_map: A mapper for block references. Programmers can use this to keep track of metadata associated to each block.
    :param dtype: The dtype to use to interpret blocks.
    :param direct: Whether to use aligned memory and disk blocks -- this is required if the file descriptors were opened using the ``O_DIRECT`` option, and can increase throughput when reads are random.
    """

    def __cinit__(self, *args, **kwargs):
        self.init_helper(*args, **kwargs)

    @binding(True)
    def init_helper(self, size_t block_size, size_t batch_size, size_t depth=32, object ref_map=None, np.dtype dtype=None, direct=DEFAULT_DIRECT):

        self.batch_size = batch_size
        self.block_size = block_size
        self.event_manager = DirectEventManager(block_size, depth=depth) if direct else EventManager(depth=depth)
        self.curr_posn = batch_size
        self.curr_refs = None
        self.curr_data = None

        self.ref_map = ref_map
        self.dtype = dtype

    @boundscheck(False)
    @wraparound(False)
    cdef int enqueue(self, int fd, size_t posn, long long ref) nogil except *:
        """
        The user needs to keep track of when enough submissions have been made to fill a batch, otherwise
        this call might release a completed batch before the user accesses it. See the example in :meth:`iter`.

        :return: 0 on success, -PERR or -ERRNO on failure.
        """

        cdef int out

        # Add a new batch if necessary
        if self.curr_posn==self.batch_size:
            out = self.event_manager.flush() # Wait for all pending writes before releasing the memory.
            if out<0:
                return out

            with gil:
                self.curr_refs = np.empty(self.batch_size, dtype=np.longlong)
                self.curr_data = np.empty((self.batch_size, self.block_size), dtype=np.uint8)
                self.curr_posn = 0

        # Enqueue the request
        out = self.event_manager.enqueue(fd, (&(self.curr_data[self.curr_posn,0])), self.block_size, posn)
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

    def iter(self, input_iter : Union[Iterable[Union[Tuple[int,int],Tuple[int,int,int]]], BaseReadInputIter]):
        """ Iterates over batches assembled from the input iterator. The input iterator can be a ``BaseReadInputIrer``-derived type (for greater efficiency),
        or a python iterable over ``(fd,posn)`` or ``(fd,posn,ref)`` tuples."""

        cdef BaseReadInputIter read_input_iter

        if isinstance(input_iter, BaseReadInputIter):
            read_input_iter = input_iter
        else:
            read_input_iter = ReadInputIterWrapper(input_iter)

        cdef int out=0
        cdef size_t k
        cdef int stop_iter = 0
        cdef FilePosn fposn

        while stop_iter == 0:

            # Build one batch
            #with nogil:
            for k in range(self.batch_size):
                fposn = read_input_iter.next()
                if fposn.fd<0:
                    stop_iter = 1
                    break
                else:
                    out = self.enqueue(fposn.fd, fposn.posn, fposn.key_id)
                    if out<0:
                        break
            if out>=0:
                out = self.event_manager.flush()


            # Yield one batch or return error
            if out < 0:
                self.do_raise(out)
            elif self.curr_data is not None and self.curr_posn > 0:
                yield self.retrieve_batch()


    def do_raise(self, int err_no):
        if self.event_manager.error_string.size()>0:
            raise Exception('Program error ' + err_str(err_no) + ': ' + self.event_manager.error_string)
        else:
            raise Exception(err_str(err_no))

from inspect import signature, Signature
def get_arg_by_name(name, *args, **kwargs):
    #NOTE: Cython discards default/keyword argument information, and hence this method
    # will only work for explicitly provided parameters.
    sgntr = signature(RAIOBatchReader.init_helper)
    return sgntr.bind_partial(None, *args, **kwargs).arguments[name]

def raio_batch_read(input_iter, *args, **kwargs):
    """
    Automatically chooses between threaded / non-threaded readers based the value of ``direct`` (a threaded reader is used when ``direct=True``).
    See :clas:`RAIOBatchRead` for a description of all arguments.
    """
    try:
        direct = get_arg_by_name('direct', *args, **kwargs)
    except KeyError:
        direct = DEFAULT_DIRECT
    if not direct:
        return raio_batch_read__non_threaded(input_iter, *args, **kwargs)
    else:
        return raio_batch_read__threaded(input_iter, *args, **kwargs)

def raio_batch_read__non_threaded(input_iter, *args, **kwargs):
    """
    Thin wrapper around :class:`RAIOBatchReader` for compatibility with the :class:`raio_batch_reader__threaded` variant below.
    """
    rbr = RAIOBatchReader(*args, **kwargs)
    return rbr.iter(input_iter)

from concurrent.futures import ThreadPoolExecutor
from pglib.parallelization.threading import ThreadOutsourcedIterable
def raio_batch_read__threaded(input_iter, *args, num_threads=2, job_queue_size=4, batches_per_job=1, **kwargs):
    """
    Creates chunks of the input iter and processes chunks using various threads. Each chunk (except possibly the last) has a multiple of batch size samples.

    Takes the same arguments as :class:`RAIOBatchReader`, as well as the keyword arguments explicit in the signature that control threads and work distribution.
    """

    cdef BaseReadInputIter read_input_iter

    job_queue_size = max(job_queue_size, num_threads)
    submitted = []
    sub_iter = [None]

    ### TODO: Use inspect to get batch_size by arg name relative to the RAIOBatchReader signature
    ### instead of using args[1] as below.
    batch_size = get_arg_by_name('batch_size', *args, **kwargs)

    if not isinstance(input_iter, BaseReadInputIter):
        read_input_iter = ReadInputIterWrapper(input_iter)
    else:
        read_input_iter = input_iter

    # TODO: Convert to cdef
    def _worker(chunk):
        return list(raio_batch_read__non_threaded(chunk, *args, **kwargs))

    # Thread-outsource chunk construction.
    with ThreadOutsourcedIterable(read_input_iter.iter_chunks(batch_size*batches_per_job)) as rii_chunks:
        rii_chunks_iter = iter(rii_chunks)
        rii_chunks_stopped = False

        # Use multiple threads to build batches.
        submitted = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            while True:
                if not rii_chunks_stopped and len(submitted) <  job_queue_size:
                    try:
                        submitted.append(executor.submit(_worker, next(rii_chunks_iter)))
                    except StopIteration:
                        rii_chunks_stopped = True
                else:
                    if not submitted:
                        break
                    batches = submitted.pop(0).result()
                    yield from batches
