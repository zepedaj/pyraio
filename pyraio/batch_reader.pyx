# distutils: language = c++

import warnings
from cpython.ref cimport PyObject, Py_XINCREF, Py_XDECREF
from cython.view cimport array as cvarray
from . cimport clibaio
from .aligned_alloc cimport aligned_alloc
from .aligned_alloc_extra cimport floor, ceil
from .util cimport buf_meta_t_2, buf_meta_t_2_str, syserr_str
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
import os

from libcpp.set cimport set as cpp_set
from libcpp.list cimport list as cpp_list
from libcpp.deque cimport deque as cpp_deque
from libcpp.vector cimport vector as cpp_vector
from libcpp cimport bool
from libc.string cimport memcpy

np.import_array()

cdef size_t ALIGN_BNDRY = 512


cdef class BlockManager:

    def __cinit__(self, size_t depth, size_t block_size):
        cdef size_t block_k
        cdef size_t max_aligned_num_bytes
        #
        self.num_submitted = 0
        self.depth = depth
        self.block_size = block_size

        # Allocating completed events memory to make it
        # possible to write to self.completed_events.data() pointer
        # even if size is 0.
        # See `NOTE: self.completed_events.data pointer was pre-allocated` below
        self.completed_events.reserve(depth)

        # Allocate block working memory
        self._blocks.resize(depth)
        self._blocks_meta.resize(depth)

        # Allocate aligned buffer space.
        max_aligned_num_bytes = floor(ALIGN_BNDRY, block_size) + 2*ALIGN_BNDRY
        self.aligned_buf_memory = aligned_alloc(ALIGN_BNDRY, max_aligned_num_bytes*depth)
        if self.aligned_buf_memory == NULL:
            raise MemoryError()

        # Prepare each block's data fields.
        for block_k in range(depth):
            clibaio.io_prep_pread(
                &(self._blocks[block_k]), -1, <char *>self.aligned_buf_memory + block_k*max_aligned_num_bytes, 0, 0)
            self._blocks[block_k].data = &self._blocks_meta[block_k]
            self._blocks_meta[block_k].ref = 0

        # Initialize unused blocks
        for block_k in range(depth):
            self.unused_blocks.push_front(&self._blocks[block_k])

        # Create io_context
        clibaio.io_setup(self.depth, &self.io_ctx)

    cdef int append_to_pending(self, int fd, size_t offset, long long ref) nogil except -1:
        # Appends a block as pending submission.
        # Returns the number of unused blocks remainig.

        cdef size_t aligned_num_bytes, aligned_offset

        if self.unused_blocks.size()==0:
            raise Exception('Attempted to append new pending block with no space left.')

        #
        aligned_offset = floor(ALIGN_BNDRY, offset)
        aligned_num_bytes = ceil(ALIGN_BNDRY, self.block_size + offset - aligned_offset)

        # Prepare new block
        iocb_p = self.unused_blocks.front()
        iocb_p[0].aio_fildes = fd
        iocb_p[0].u.c.nbytes = aligned_num_bytes;
        iocb_p[0].u.c.offset = aligned_offset;
        self.unused_blocks.pop_front()

        # Prepare block's meta
        meta_p = <buf_meta_t_2 *>iocb_p[0].data
        meta_p[0].data_start = offset - aligned_offset
        meta_p[0].data_end = offset + self.block_size - aligned_offset
        meta_p[0].fd = fd
        meta_p[0].offset = offset
        meta_p[0].num_bytes = self.block_size
        meta_p[0].ref = ref

        # Append block
        self.pending_blocks.push_back(iocb_p)

    cdef int submit_pending(self) nogil except -1:
        # Submits all pending events.

        cdef int num_submitted_or_error
        num_submitted_or_error = clibaio.io_submit(self.io_ctx, self.pending_blocks.size(), self.pending_blocks.data())
        self.num_submitted += max(0,num_submitted_or_error)

        if num_submitted_or_error < 0:
            with gil: raise Exception(f'Error {syserr_str(num_submitted_or_error)} when attempting to submit blocks.')

        if <size_t>num_submitted_or_error != self.pending_blocks.size():
            with gil: raise Exception(f'Blocks submitted {num_submitted_or_error} do not match requested number {self.pending_blocks.size()}.')

        # Remove all pending
        self.pending_blocks.resize(0)

    cdef int get_completed(self, long min_nr) nogil except -1:
        """
        :param min_nr: The min number of events to retrieve. Use a non-positive value to retrieve all submitted.
        """

        cdef int num_retrieved

        if self.completed_events.size() != 0:
            raise Exception('You need to release the completed events before calling `get_completed` a second time.')

        #
        if min_nr <= 0:
            min_nr = self.num_submitted
        elif min_nr > self.num_submitted:
            with gil: raise Exception("Attempted to retrieve more events than were submitted.")

        # Get completed blocks.
        if min_nr >0:
            # NOTE: self.completed_events.data pointer was pre-allocated.
            self.completed_events.resize(self.depth)
            num_retrieved = clibaio.io_getevents(self.io_ctx, min_nr, self.depth, self.completed_events.data(), NULL)
            self.completed_events.resize(max(0,num_retrieved))
            if num_retrieved>0:
                self.num_submitted -= max(0,num_retrieved)
            elif num_retrieved<0:
                with gil: raise Exception(f'Error occurred when attempting to get events ({num_retrieved}).')

    cdef int release_completed(self) nogil except -1:
        cdef size_t k_event
        for k_event in range(self.completed_events.size()):
            self.unused_blocks.insert(self.unused_blocks.begin(), self.completed_events[k_event].obj)
        self.completed_events.resize(0)

    def __dealloc__(self):

        cdef int num_retrieved

        # Wait for pending events.
        if self.num_submitted>0:
            # TODO: NULL timespec (last param) could lead to hang.
            num_retrieved = clibaio.io_getevents(self.io_ctx, self.num_submitted, self.depth, self.completed_events.data(), NULL)
            if self.num_submitted != num_retrieved:
                warnings.warn("Failed to get pending events in preparation for memory deallocation. Will attempt to free memory nonetheless.")

        # Destroy the io context
        if self.io_ctx:
            clibaio.io_destroy(self.io_ctx)

        # Release block buffer memory
        free(self.aligned_buf_memory)


cdef class RAIOBatchReader:

    def __cinit__(self, size_t block_size, size_t batch_size, size_t depth=32, object ref_map=None, np.dtype dtype=None, tuple shape=None):

        self.batch_size = batch_size
        self.block_manager = BlockManager(depth, block_size)
        self.curr_posn = batch_size
        self.batches = []

        self.ref_map = ref_map
        self.dtype = dtype
        self.shape = shape

    #TODO: @boundscheck(False)
    cdef int submit(self, long fd, size_t posn, long long ref) nogil except -1:
        """
        Adds a new request to the pending blocks. If the pending blocks become full, the call gets the available completed events (or waits for
        at least one to become available) and writes them to the ``batches``.

        Use a negative ``fd`` will flush all buffers: all pending requests are submitted and the call blocks until all submitted requests are completed and written to ``batches``.

        Will return a 1 if a new batch was initialized and a 0 otherwise. Note that the last batch in :attr:`batches` might not be completed, and only
        :attr:`curr_posn` elements have been written.
        """
        cdef size_t k
        cdef char[:,:] out_buffer = None
        cdef int new_batch
        cdef clibaio.io_event event

        # Append a new block, exit if the submission queue is not full.
        if fd > 0:
            self.block_manager.append_to_pending(fd, posn, ref)
            if self.block_manager.unused_blocks.size()!=0:
                # Wait for the submission queue to be full
                # before submitting more blocks.
                return 0

        # Submit a full depth worth of blocks and get at least one.
        self.block_manager.submit_pending()
        self.block_manager.get_completed(-1 if fd < 0 else 1)
        new_batch = self.write_completed()
        self.block_manager.release_completed()

        return new_batch

    cdef int write_completed(self) nogil except -1:
        """ Writes all completed events to the output batch and returns an integer indicating whether the batch is complete."""

        cdef int new_batch = 0
        cdef size_t k_event

        # Copy blocks to output batch.
        for k_event in range(self.block_manager.completed_events.size()):

            event_p = &self.block_manager.completed_events[k_event]

            # Add a new batch if necessary
            if self.curr_posn==self.batch_size:
                with gil:
                    self.curr_refs = np.empty(self.batch_size, dtype=np.longlong)
                    self.curr_data = np.empty((self.batch_size, self.block_manager.block_size), dtype=np.uint8)
                    self.batches.append((self.curr_refs, self.curr_data))
                    new_batch = 1
                self.curr_posn = 0

            # Get next sample
            res = <long int>(event_p[0].res)
            res2 = <long int>(event_p[0].res2)
            iocb_p = (event_p[0].obj)
            block_meta_p = <buf_meta_t_2 *>(iocb_p[0].data)
            buf_ptr = iocb_p[0].u.c.buf
            nbytes = iocb_p[0].u.c.nbytes

            # Check results
            if res<0 or res2 != 0:
                with gil: raise Exception(f'Error res={syserr_str(res)}, res2={syserr_str(res2)} with retrieved event for request {buf_meta_t_2_str(block_meta_p[0])}.')
            elif <size_t>res < block_meta_p[0].data_end:
                with gil: raise Exception(f'Failed to read the requested number of bytes. Read {res} bytes but required {block_meta_p[0].data_end} for request {buf_meta_t_2_str(block_meta_p[0])}.')

            # Update output batch
            self.curr_refs[self.curr_posn] = (<buf_meta_t_2 *>(event_p[0].data))[0].ref
            memcpy(&(self.curr_data[self.curr_posn,0]), <char *>buf_ptr + block_meta_p[0].data_start, self.block_manager.block_size)
            self.curr_posn += 1

        return new_batch


    def format_batch(self, long long[:] refs, char[:,:] data, int prune):

        if prune>=0:
            refs = refs[:prune]
            data = data[:prune, :]

        #
        if self.dtype is not None:
            out_data = data.view(self.dtype)
        else:
            out_data = data

        #
        if self.ref_map is not None:
            out_refs = [self.ref_map[k] for k in refs]
        else:
            out_refs = refs

        return out_refs, out_data

    def iter(self, input_iter, long long default_ref=0, ref_map=None, dtype=None, shape=None):
        """
        Random Acess IO read by batches. Reads randomly positioned parts of one or more files using low-level system support for read IO parallelization.
        The heavy-duty computations in this class release the GIL -- running this iterator in a separate thread will hence enjoy greater efficiency.

        :param block_iter: An iterator that produces tuples of the form  ``(<int file descriptor>, <int offset>)`` ``(<int file descriptor>, <int offset>, <int ref>)``.
        The file descriptors should be obtained using :func:`~pyraio.util.raio_open` or :func:`~pyraio.util.raio_open_ctx`.
        :param block_size: The size of read blocks.
        :param out_buf_iter: Data is written sequentially to the buffers supplied by this iterator. Each supplied buffer must have a size that is a multiple of ``block_size``.
        :param depth: The number of simultaneous I/O requests submitted to the OS.
        :param with_refs: Whether to gather and yield references together with the output buffers.
        :return: Yields each buffer from ``out_buf_iter`` once it has been filled with data. The last buffer yielded might be a slice of the provided buffer if there are less
            samples in ``block_iter`` than required to fill that last buffer. Together, with each buffer, yields a list of the references that correspond to the data in the buffer
            (unless ``with_refs=False``).


        .. testcode::

            from pyraio import raio_batch_read, raio_open_ctx
            from tempfile import NamedTemporaryFile
            import numpy as np

            with NamedTemporaryFile(mode="wb") as fo:

                # Write some data to the file.
                N = int(1e6)
                rng = np.random.default_rng()
                dat = rng.integers(256, size=N).astype("u1")
                fo.write(dat)
                fo.flush()

                # An iterator that produces output buffers
                def out_batch_iter(size):
                    while True:
                        yield np.empty(size, dtype="u1")

                # Read the data, 700 bytes at a time
                num_bytes = 700
                batch_size = 5  # Will return 5 samples of 700 bytes at a time.
                num_batches = 4  # Read four batches
                offsets = range(
                    0, num_bytes * batch_size * num_batches, num_bytes
                )  # Read four batches

                with raio_open_ctx(fo.name) as fd:
                    dat = [
                        batch
                        for batch, ref in raio_batch_read(
                            ((fd, offset, None) for offset in offsets),
                            num_bytes,
                            out_batch_iter(num_bytes * batch_size),
                        )
                    ]

                assert [len(_x) for _x in dat] == [batch_size * num_bytes] * num_batches

        """
        cdef long fd
        cdef size_t posn
        cdef long long ref=default_ref
        cdef int k_batch, skip_last

        while True:
            try:
                vals = next(input_iter)
            except StopIteration:
                fd, posn = -1, 0
            else:
                if len(vals)==2:
                    fd, posn = vals
                elif len(vals)==3:
                    fd, posn, ref = vals
                else:
                    raise Exception(f'Expected 2 or 3 values but obtained {len(vals)}.')

            self.submit(fd, posn, ref)

            skip_last = 1 if fd>=0 and self.curr_posn<self.batch_size else 0
            for k_batch in range(len(self.batches)-skip_last):
                out_batch = self.batches.pop(0)
                out = self.format_batch(*out_batch, self.curr_posn if len(self.batches)==0 else -1)
                yield out

            if fd < 0:
                break


def raio_batch_read(input_iter, block_size, batch_size, depth=32, ref_map=None, dtype=None):
    rbr = RAIOBatchReader(block_size, batch_size, depth, ref_map=ref_map, dtype=dtype)
    yield from rbr.iter(input_iter)
