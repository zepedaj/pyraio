# distutils: language = c++

from libc.stdio cimport printf
cimport clibaio
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

from libcpp.set cimport set as cpp_set
from libcpp.list cimport list as cpp_list
from libcpp cimport bool

np.import_array()


cdef size_t prepare_blocks_to_submit(block_iter, np_bufs, cpp_list[clibaio.iocb *] &unused_blocks, clibaio.iocb **&blocks_to_submit):

    cdef bool iter_exhausted = False
    cdef size_t block_k = 0
    cdef char[:] buf_mview
    cdef void * buf_voidp
    cdef int fd
    cdef size_t num_bytes
    cdef long long offset

    # Prepare new io requests
    while not iter_exhausted and unused_blocks.size()>0:
        try:
            fd, offset, num_bytes = next(block_iter)
        except StopIteration:
            iter_exhausted = True
            break

        # Prepare new numpy array
        np_arr = np.empty(num_bytes, dtype=np.uint8)
        buf_mview = np_arr
        buf_voidp = &buf_mview[0]
        np_bufs[<long int>buf_voidp] = np_arr

        # Prepare new block
        iocb_p = unused_blocks.front()
        clibaio.io_prep_pread(
           iocb_p, fd, buf_voidp, num_bytes, offset)
        unused_blocks.pop_front()
        blocks_to_submit[block_k] = iocb_p

        block_k+=1

    return block_k


def read_blocks(block_iter, size_t max_events=32):

    # Create blocks
    cdef clibaio.iocb *blocks_memory
    blocks_memory = <clibaio.iocb *> malloc(max_events * sizeof(clibaio.iocb))
    if not blocks_memory:
        raise MemoryError()

    cdef clibaio.iocb **blocks_to_submit
    blocks_to_submit = <clibaio.iocb **> malloc(max_events * sizeof(clibaio.iocb *))
    if not blocks_to_submit:
        free(blocks_memory)
        raise MemoryError()

    cdef size_t num_completed_events = 0
    cdef clibaio.io_event *completed_events
    cdef clibaio.io_event *next_completed_event=NULL
    completed_events = <clibaio.io_event *> malloc(max_events * sizeof(clibaio.io_event))
    if not blocks_to_submit:
        free(blocks_memory)
        free(blocks_to_submit)
        raise MemoryError()

    cdef cpp_list[clibaio.iocb *] unused_blocks
    cdef clibaio.io_context_t io_ctx
    cdef size_t block_k, num_to_submit
    cdef bool iter_exhausted=False

    cdef char[:] buf_mview
    cdef void * buf_voidp

    cdef clibaio.iocb * iocb_p


    np_bufs = {}

    try:

        # Set unused blocks to all available
        for block_k in range(max_events):
            unused_blocks.push_front(blocks_memory+block_k)

        # Create io_context
        clibaio.io_setup(max_events, &io_ctx)

        try:
            while True:
                # Prepare new io requests
                num_to_submit = prepare_blocks_to_submit(block_iter, np_bufs, unused_blocks, blocks_to_submit)

                # Submit new io requests
                if num_to_submit>0:
                    num_submitted = <size_t>clibaio.io_submit(io_ctx, num_to_submit, blocks_to_submit)
                    if num_submitted != num_to_submit:
                        raise Exception(f'Blocks submitted {num_submitted} to not match requested number {num_to_submit}.')
                elif unused_blocks.size()==max_events:
                    # Finished all computations.
                    return

                # Get completed requests, if none are available.
                if num_completed_events == 0:
                    num_completed_events = clibaio.io_getevents(io_ctx, 1, max_events, completed_events, NULL)
                    next_completed_event = completed_events

                # Yield buffer
                res = <long int>next_completed_event[0].res
                res2 = <long int>next_completed_event[0].res2
                if res<0 or res2 != 0:
                    raise Exception(f'Failed event with res={res} and res2={res2}.')
                iocb_p = next_completed_event[0].obj
                buf_id = iocb_p[0].u.c.buf
                unused_blocks.push_front(iocb_p)
                yield np_bufs.pop(<long int>buf_id)

                # Move to next event
                num_completed_events -=1
                next_completed_event += 1

        finally:
            clibaio.io_destroy(io_ctx)
    finally:
        free(blocks_memory)
        free(blocks_to_submit)
        free(completed_events)
