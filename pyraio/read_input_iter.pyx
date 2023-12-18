# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8


cdef class BaseReadInputIter:
    """
    Classes can implement this interface to benefit from faster speeds when using :meth:`BatchReader.iter`.
    """
    cdef FilePosn next(self) noexcept nogil:
        with gil:
            raise NotImplementedError('Abstract method.')

    def iter_chunks(self, int chunk_size):

        cdef ReadInputIterChunk chunk

        while True:
            chunk = ReadInputIterChunk()
            with nogil:
                chunk.populate(self, chunk_size)
            if chunk.file_posns.size()==0:
                break
            else:
                yield chunk

    def __iter__(self):
        while True:
            fl_posn = self.next()
            if fl_posn.fd == -1:
                return
            else:
                yield fl_posn.fd, fl_posn.posn, fl_posn.key_id


cdef class ReadInputIterChunk(BaseReadInputIter):
    """ A chunk from a :class:`BaseReadInputIter` object that behaves in the same way and is used to parallelize random reads across threads."""

    cdef void populate(self, BaseReadInputIter read_input_iter, int chunk_size) noexcept nogil:

        cdef int curr_posn=0
        self.file_posns.resize(chunk_size)
        while curr_posn < chunk_size:
            self.file_posns[curr_posn] = read_input_iter.next()
            if self.file_posns[curr_posn].fd<0:
                self.file_posns.resize(curr_posn)
                break
            else:
                curr_posn+=1
        self.posn = 0

    cdef FilePosn next(self) noexcept nogil:
        cdef FilePosn out
        if <size_t>self.posn>=self.file_posns.size():
            out.fd = -1
        else:
            out = self.file_posns[self.posn]
            self.posn+=1
        return out

cdef class ReadInputIterWrapper(BaseReadInputIter):
    """
    Wraps a generic python iterable as a :class:`BaseReadInputIter`.
    """

    def __cinit__(self, iterator, default_ref=0):
        self.source_iterator = iter(iterator)
        self.default_ref = 0

    cdef FilePosn next(self) noexcept nogil:

        cdef FilePosn fl_posn

        with gil:
            try:
                vals = next(self.source_iterator)
            except StopIteration:
                fl_posn.fd = -1
            else:
                if len(vals)==2:
                    fl_posn.fd, fl_posn.posn = vals
                    fl_posn.key_id = self.default_ref
                elif len(vals)==3:
                    fl_posn.fd, fl_posn.posn, fl_posn.key_id = vals
                else:
                    raise Exception(f'Expected 2 or 3 values but obtained {len(vals)}.')

                if fl_posn.fd < 0:
                    raise OverflowError(f'Invalid negative fd={vals[0]}')
                if fl_posn.posn < 0:
                    raise OverflowError(f'Invalid negative posn={vals[1]}')

        return fl_posn
