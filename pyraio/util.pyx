# distutils: language = c++

import os
from contextlib import contextmanager
import errno

cdef str buf_meta_t_str(buf_meta_t &buf_meta):
    filename = os.readlink(f'/proc/self/fd/{buf_meta.fd}')
    return f"<{filename}, offset={buf_meta.offset}, num_bytes={buf_meta.num_bytes} | data_start={buf_meta.data_start}, data_end={buf_meta.data_end}>"

cdef str syserr_str(int err_num):
    return errno.errorcode.get(abs(err_num), f"<UNKNOWN:{err_num}>")

cpdef raio_open(str filename):
    """
    Opens the file for reading. The file mode will include the ``os.O_DIRECT`` flag required by :func:`raio_read`.

    :param filename: The filename or path.
    :return: The file descriptor.

    .. todo:: This function might support alternate modes to support possible ``raio_write``  and ``raio_read_write``.
    """
    return os.open(str(filename), os.O_DIRECT | os.O_RDONLY)

@contextmanager
def raio_open_ctx(filename):
    """
    A context manager for :func:`raio_open`.
    """
    fd = raio_open(filename)
    yield fd
    os.close(fd)
