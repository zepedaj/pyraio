# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8
#
"""
TODO:
1. Documentation
2. Make sure all cdef html functions are completely white
3. Move event managers to new pyx/pxd file
4. Enable keyboard interrupts in random VanilaDataset iterations. 
"""

from typing import Union, Iterable, Tuple
from .read_input_iter cimport FilePosn, ReadInputIterWrapper, BaseReadInputIter
from cython cimport wraparound, boundscheck, binding
import numpy as np
cimport numpy as np
from libcpp cimport bool as cbool
from contextlib import nullcontext
from .event_managers cimport DirectEventManager, EventManager
from .event_managers import PERR, err_str
from .event_managers cimport PERR

np.import_array()

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
    :param direct: Whether to use aligned memory and aligned disk blocks -- this is required if the file descriptors were opened using the ``O_DIRECT`` option, and can increase throughput when reads are random.
    :param drop_gil: This must be set to ``True`` if using threading (e.g., through a call to :meth:`raio_batch_read__threaded`) -- doing so builds batches using a ``with nogil`` block. If threading is not used, making this ``False`` results in faster execution.   
    """

    def __cinit__(self, *args, **kwargs):
        self.init_helper(*args, **kwargs)

    @binding(True)
    def init_helper(self, size_t block_size, size_t batch_size, size_t depth=32, object ref_map=None, np.dtype dtype=None, cbool direct=DEFAULT_DIRECT, drop_gil=None):        

        self.batch_size = batch_size
        self.block_size = block_size
        self.event_manager = DirectEventManager(block_size, depth=depth) if direct else EventManager(depth=depth)
        self.drop_gil = drop_gil if drop_gil is not None else direct

        self.ref_map = ref_map
        self.dtype = dtype

    def format_batch(self, refs, data):

        refs = np.asarray(refs)
        data = np.asarray(data)

        #
        if self.dtype is not None:
            data = data.view(self.dtype)

        #
        if self.ref_map is not None:
            refs = [self.ref_map[k] for k in refs]

        return refs, data

    @boundscheck(False)
    @wraparound(False)
    cdef int build_batch(self, BaseReadInputIter read_input_iter, long long[:] refs, char[:,:] data) nogil except *:
        cdef size_t block_count=0, k=0
        cdef int out=0
        cdef FilePosn fposn
        
        for k in range(self.batch_size):
            fposn = read_input_iter.next()            
            if fposn.fd<0:
                break
            else:
                # Udate data
                out = self.event_manager.enqueue(fposn.fd, &(data[k,0]), self.block_size, fposn.posn)
                if out<0:
                    return out

                # Update refs
                refs[k] = fposn.key_id
                block_count+=1

        out = self.event_manager.flush()
        if out<0:
            return out
        
        return block_count
        
    
    def iter(self, input_iter : Union[Iterable[Union[Tuple[int,int],Tuple[int,int,int]]], BaseReadInputIter]):
        """ Iterates over batches assembled from the input iterator. The input iterator can be a ``BaseReadInputIrer``-derived type (for greater efficiency),
        or a python iterable over ``(fd,posn)`` or ``(fd,posn,ref)`` tuples."""

        cdef BaseReadInputIter read_input_iter
        cdef long long[:] refs = None
        cdef char[:,:] data = None

        if isinstance(input_iter, BaseReadInputIter):
            read_input_iter = input_iter
        else:
            read_input_iter = ReadInputIterWrapper(input_iter)

        cdef long long block_count=self.batch_size       
                

        while block_count==<long long>self.batch_size:
            
            # Build one batch
            refs = np.empty(self.batch_size, dtype=np.longlong)
            data = np.empty((self.batch_size, self.block_size), dtype=np.uint8)

            if self.drop_gil:
                with nogil:            
                    block_count = self.build_batch(read_input_iter, refs, data)
            else:
                block_count = self.build_batch(read_input_iter, refs, data)

            # Yield one batch or return error
            if block_count < 0:
                self.do_raise(block_count)
            elif block_count > 0:
                yield self.format_batch(refs[:block_count], data[:block_count, :])

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
