# distutils: language = c++

from cython.view cimport array as cvarray
from libc.stdio cimport printf
cimport clibaio
from aligned_alloc cimport aligned_alloc
from aligned_alloc_extra cimport floor, ceil
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

from libcpp.set cimport set as cpp_set
from libcpp.list cimport list as cpp_list
from libcpp cimport bool

np.import_array()

cdef size_t ALIGN_BNDRY = 512


ctypedef struct buf_meta_t:
    size_t data_start # Data start position in aligned buffer
    size_t data_end # Data end position in aligned buffer


cdef void free_aligned(void *ptr):
    free(<void *>floor(ALIGN_BNDRY, <size_t>ptr))

cdef size_t prepare_blocks_to_submit(block_iter, cpp_list[clibaio.iocb *] &unused_blocks, clibaio.iocb **&blocks_to_submit):

    cdef size_t block_k = 0
    cdef char[:] buf_mview
    cdef void * buf_voidp
    cdef int fd
    cdef size_t num_bytes, aligned_num_bytes
    cdef long long offset, aligned_offset
    cdef buf_meta_t *meta_p

    # Prepare new io requests
    while unused_blocks.size()>0:
        try:
            fd, offset, num_bytes = next(block_iter)
        except StopIteration:
            return block_k

        # Prepare new memory
        aligned_offset = floor(ALIGN_BNDRY, offset)
        aligned_num_bytes = ceil(ALIGN_BNDRY, num_bytes + offset - aligned_offset)
        buf_voidp = aligned_alloc(ALIGN_BNDRY, aligned_num_bytes)

        # Prepare new block
        iocb_p = unused_blocks.front()
        meta_p = <buf_meta_t *>iocb_p[0].data # Will be cleared by io_prep_pread -- must read before.
        clibaio.io_prep_pread(
           iocb_p, fd, buf_voidp, aligned_num_bytes, aligned_offset)
        unused_blocks.pop_front()

        # Add block meta data.
        iocb_p[0].data = meta_p
        meta_p[0].data_start = offset - aligned_offset
        meta_p[0].data_end = offset + num_bytes - aligned_offset

        # Append block
        blocks_to_submit[block_k] = iocb_p

        block_k+=1

    return block_k


def read_blocks(block_iter, size_t max_events=32):

    cdef size_t block_k, num_to_submit

    # Create blocks
    cdef clibaio.iocb *blocks_memory
    blocks_memory = <clibaio.iocb *> malloc(max_events * sizeof(clibaio.iocb))
    if not blocks_memory:
        raise MemoryError()

    # Buffer meta memory
    cdef  buf_meta_t *buf_meta_memory
    buf_meta_memory = <buf_meta_t *> malloc(max_events * sizeof(buf_meta_t))
    cdef buf_meta_t block_meta
    if not blocks_memory:
        free(blocks_memory)
        raise MemoryError()
    for block_k in range(max_events):
        blocks_memory[block_k].data = buf_meta_memory + block_k

    cdef clibaio.iocb **blocks_to_submit
    blocks_to_submit = <clibaio.iocb **> malloc(max_events * sizeof(clibaio.iocb *))
    if not blocks_to_submit:
        free(blocks_memory)
        free(buf_meta_memory)
        raise MemoryError()

    cdef size_t num_completed_events = 0
    cdef clibaio.io_event *completed_events
    cdef clibaio.io_event *next_completed_event=NULL
    completed_events = <clibaio.io_event *> malloc(max_events * sizeof(clibaio.io_event))
    if not blocks_to_submit:
        free(blocks_memory)
        free(buf_meta_memory)
        free(blocks_to_submit)
        raise MemoryError()

    cdef cpp_list[clibaio.iocb *] unused_blocks
    cdef clibaio.io_context_t io_ctx

    cdef clibaio.iocb * iocb_p

    cdef cvarray buf_arr

    try:

        # Set unused blocks to all available
        for block_k in range(max_events):
            unused_blocks.push_front(blocks_memory+block_k)

        # Create io_context
        clibaio.io_setup(max_events, &io_ctx)

        try:
            while True:
                # Prepare new io requests
                num_to_submit = prepare_blocks_to_submit(block_iter, unused_blocks, blocks_to_submit)

                # Submit new io requests
                if num_to_submit>0:
                    num_submitted = <size_t>clibaio.io_submit(io_ctx, num_to_submit, blocks_to_submit)
                    if num_submitted != num_to_submit:
                        raise Exception(f'Blocks submitted {num_submitted} do not match requested number {num_to_submit}.')
                elif unused_blocks.size()==max_events:
                    # Finished all computations.
                    return

                # Get completed requests, if none are available.
                if num_completed_events == 0:
                    num_completed_events = clibaio.io_getevents(io_ctx, 1, max_events, completed_events, NULL)
                    if num_completed_events<0:
                        raise Exception(f'Error occurred when attempting to get events ({num_completed_events}).')
                    next_completed_event = completed_events

                # Yield buffer
                res = <long int>next_completed_event[0].res
                res2 = <long int>next_completed_event[0].res2
                if res<0 or res2 != 0:
                    raise Exception(f'Failed event with res={res} and res2={res2}.')
                iocb_p = next_completed_event[0].obj
                buf_ptr = iocb_p[0].u.c.buf
                nbytes = iocb_p[0].u.c.nbytes
                if res<0 or <unsigned long>res != nbytes:
                    raise Exception('Failed to read the requested number of bytes!')
                unused_blocks.push_front(iocb_p)

                # Convert buffer to numpy object
                block_meta = (<buf_meta_t *>iocb_p[0].data)[0]
                buf_arr = <char[:(block_meta.data_end - block_meta.data_start)]> (<char*>buf_ptr+block_meta.data_start)
                buf_arr.callback_free_data = free_aligned
                yield buf_arr

                # Move to next event
                num_completed_events -=1
                next_completed_event += 1

        finally:
            clibaio.io_destroy(io_ctx)
    finally:
        free(blocks_memory)
        free(buf_meta_memory)
        free(blocks_to_submit)
        free(completed_events)
