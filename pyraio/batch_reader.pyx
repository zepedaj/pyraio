# distutils: language = c++
import warnings
from cpython.ref cimport PyObject, Py_XINCREF, Py_XDECREF
from cython.view cimport array as cvarray
from . cimport clibaio
from .aligned_alloc cimport aligned_alloc
from .aligned_alloc_extra cimport floor, ceil
from .util cimport buf_meta_t, buf_meta_t_str, syserr_str
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
import os

from libcpp.set cimport set as cpp_set
from libcpp.list cimport list as cpp_list
from libcpp cimport bool
from libc.string cimport memcpy

np.import_array()

cdef size_t ALIGN_BNDRY = 512

cdef size_t prepare_blocks_to_submit(block_iter, size_t num_bytes, cpp_list[clibaio.iocb *] &unused_blocks, clibaio.iocb **&blocks_to_submit) except -1:

    cdef size_t block_k = 0
    cdef int fd
    cdef size_t aligned_num_bytes
    cdef size_t offset, aligned_offset
    cdef buf_meta_t *meta_p

    # Prepare new io requests
    while unused_blocks.size()>0:
        try:
            fd, offset, ref = next(block_iter)
        except StopIteration:
            return block_k

        #
        aligned_offset = floor(ALIGN_BNDRY, offset)
        aligned_num_bytes = ceil(ALIGN_BNDRY, num_bytes + offset - aligned_offset)

        # Prepare new block
        iocb_p = unused_blocks.front()
        iocb_p[0].aio_fildes = fd
        iocb_p[0].u.c.nbytes = aligned_num_bytes;
        iocb_p[0].u.c.offset = aligned_offset;
        unused_blocks.pop_front()

        # Prepare block's meta
        meta_p = <buf_meta_t *>iocb_p[0].data
        meta_p[0].data_start = offset - aligned_offset
        meta_p[0].data_end = offset + num_bytes - aligned_offset
        meta_p[0].fd = fd
        meta_p[0].offset = offset
        meta_p[0].num_bytes = num_bytes
        # TODO: These next two statements should be atomic, but an interrupt can prevent that.
        meta_p[0].ref = <PyObject *>ref
        Py_XINCREF(meta_p[0].ref)

        # Append block
        blocks_to_submit[block_k] = iocb_p

        block_k+=1

    return block_k

def raio_batch_read(block_iter, size_t block_size, out_buf_iter, size_t depth=32, bool with_refs = True):
    """
    raio_batch_read(block_iter, size_t block_size, out_buf_iter, size_t depth=32, bool with_refs = True)

    Random Acess IO read by batches. Reads randomly positioned parts of one or more files using low-level system support for parallelization without the need for threads.

    :param block_iter: An iterator that produces tuples of the form ``(<file descriptor>, <offset>, <ref>)``. The file descriptors should be obtained using :func:`~pyraio.util.raio_open` or :func:`~pyraio.util.raio_open_ctx`.
        Each block read will have size ``block_size``.
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

    cdef size_t block_k, num_to_submit
    cdef size_t num_bytes_to_copy=0, num_bytes_copied=0

    cdef list out_ref_list

    # Memory that needs to be freed
    cdef clibaio.iocb *blocks_memory = NULL
    cdef void *block_buf_memory = NULL
    cdef  buf_meta_t *buf_meta_memory = NULL
    cdef clibaio.iocb **blocks_to_submit = NULL
    cdef clibaio.io_event *completed_events = NULL

    # Helper pointers
    cdef clibaio.io_event *next_completed_event=NULL
    cdef cpp_list[clibaio.iocb *] unused_blocks
    cdef clibaio.io_context_t io_ctx = NULL
    cdef clibaio.iocb * iocb_p

    cdef char[:] out_buf
    cdef char *out_buf_ptr=NULL
    cdef char *out_buf_end=NULL

    cdef size_t max_aligned_num_bytes = floor(ALIGN_BNDRY, block_size) + 2*ALIGN_BNDRY
    cdef buf_meta_t *block_meta_p
    cdef size_t num_completed_events = 0, num_pending_events = 0

    try:


        # Create block memory
        blocks_memory = <clibaio.iocb *> malloc(depth * sizeof(clibaio.iocb))
        if not blocks_memory:
            raise MemoryError()

        # Create block buffer memory
        block_buf_memory = aligned_alloc(ALIGN_BNDRY, max_aligned_num_bytes*depth)
        if not block_buf_memory:
            raise MemoryError()

        # Create buffer meta memory
        buf_meta_memory = <buf_meta_t *> malloc(depth * sizeof(buf_meta_t))
        if not blocks_memory:
            raise MemoryError()


        # Prepare each block's data fields.
        for block_k in range(depth):
            clibaio.io_prep_pread(
                blocks_memory+block_k, -1, <char *>block_buf_memory + block_k*max_aligned_num_bytes, 0, 0)
            blocks_memory[block_k].data = buf_meta_memory+block_k
            buf_meta_memory[block_k].ref = NULL


        # Create blocks to submit memory
        blocks_to_submit = <clibaio.iocb **> malloc(depth * sizeof(clibaio.iocb *))
        if not blocks_to_submit:
            raise MemoryError()

        # Create completed events
        completed_events = <clibaio.io_event *> malloc(depth * sizeof(clibaio.io_event))
        if not blocks_to_submit:
            raise MemoryError()

        # Set unused blocks to all available
        for block_k in range(depth):
            unused_blocks.push_front(blocks_memory+block_k)

        # Create io_context
        clibaio.io_setup(depth, &io_ctx)

        while True:
            # Prepare new io requests
            num_to_submit = prepare_blocks_to_submit(block_iter, block_size, unused_blocks, blocks_to_submit)

            # Submit new io requests
            if num_to_submit>0:
                num_submitted=0

                # TODO: What happens when you submit and some but not all blocks fail? Will the response be negative?
                # Will num_pending_events have the wrong value? If so, possible seg fault if blocks are released while
                # libaio is still writing to them.
                # TODO: These next two statements should be atomic, but an interrupt can prevent that.
                with nogil: num_submitted = clibaio.io_submit(io_ctx, num_to_submit, blocks_to_submit)
                num_pending_events += max(0,num_submitted)

                if num_submitted <0:
                    raise Exception(f'Error {syserr_str(num_submitted)} when attempting to submit blocks.')
                if <size_t>num_submitted != num_to_submit:
                    raise Exception(f'Blocks submitted {num_submitted} do not match requested number {num_to_submit}.')
            elif unused_blocks.size()==depth and num_completed_events == 0:
                # Finished all computations.

                # Yield last batch
                if out_buf_iter is not None and out_buf_ptr != NULL:
                    yield out_buf[:-(out_buf_end-out_buf_ptr) or None], (out_ref_list if with_refs else None)

                # Done
                return

            # Get completed requests, if none are available.
            if num_completed_events == 0:

                # TODO: These next two statements should be atomic, but an interrupt can prevent that.
                with nogil: num_completed_events = clibaio.io_getevents(io_ctx, 1, depth, completed_events, NULL)
                num_pending_events -= num_completed_events

                if num_completed_events<0:
                    raise Exception(f'Error occurred when attempting to get events ({num_completed_events}).')
                next_completed_event = completed_events


            # Get next sample
            res = <long int>next_completed_event[0].res
            res2 = <long int>next_completed_event[0].res2
            iocb_p = next_completed_event[0].obj
            block_meta_p = (<buf_meta_t *>iocb_p[0].data)
            buf_ptr = iocb_p[0].u.c.buf
            nbytes = iocb_p[0].u.c.nbytes

            if res<0 or res2 != 0:
                raise Exception(f'Error res={syserr_str(res)}, res2={syserr_str(res2)} with retrieved event for request {buf_meta_t_str(block_meta_p[0])}.')
            elif <size_t>res < block_meta_p[0].data_end:
                raise Exception(f'Failed to read the requested number of bytes. Read {res} bytes but required {block_meta_p[0].data_end} for request {buf_meta_t_str(block_meta_p[0])}.')
            unused_blocks.push_front(iocb_p)

            # Get the next output buffer
            if out_buf_ptr == out_buf_end:
                try:
                    out_buf = next(out_buf_iter)
                    out_ref_list = []
                    if len(out_buf)%block_size:
                        raise Exception('The output buffer lengths must be multiples of the block size.')
                except StopIteration:
                    # Ran out of output buffers.
                    raise Exception('The output buffers iterator stopped before the blocks iterator.')

                if len(out_buf)==0:
                    raise Exception('Output buffers cannot be size 0!')
                else:
                    out_buf_ptr = &out_buf[0]
                    out_buf_end = &out_buf[-1]+1


            # Copy the sample to the output buffer.
            with nogil: memcpy(out_buf_ptr, <char *>buf_ptr + block_meta_p[0].data_start, block_size)
            if with_refs:
                out_ref_list.append(<object>block_meta_p[0].ref)
            out_buf_ptr += block_size

            # Yield the output buffer
            if out_buf_ptr == out_buf_end:
                # The output buffer is full, yield it.
                out_buf_ptr = NULL
                out_buf_end = NULL
                yield out_buf, (out_ref_list if with_refs else None)
                out_ref_list = None


            # TODO: These next two statements should be atomic, but an interrupt can prevent that.
            Py_XDECREF(block_meta_p[0].ref)
            block_meta_p[0].ref = NULL

            # Move to next event
            num_completed_events -=1
            next_completed_event += 1

    finally:
        # Wait for pending events.
        if num_pending_events>0:
            # TODO: NULL timespec (last param) could lead to hang.
            num_completed_events = clibaio.io_getevents(io_ctx, num_pending_events, depth, completed_events, NULL)
            if num_pending_events != num_completed_events:
                warnings.warn("Failed to get pending events in preparation for memory deallocation. Will attempt to free memory nonetheless.")

        # Destroy the io context
        if io_ctx:
            clibaio.io_destroy(io_ctx)

        # Decrease all internally held references not yet decreased.
        for block_k in range(depth):
            Py_XDECREF((<buf_meta_t *>(blocks_memory[block_k].data))[0].ref)

        # Free memory
        free(blocks_memory)
        free(block_buf_memory)
        free(buf_meta_memory)
        free(blocks_to_submit)
        free(completed_events)
