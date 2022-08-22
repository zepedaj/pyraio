import pyraio as mdl
import re
import pytest
import numpy.testing as npt
from itertools import chain
from tempfile import NamedTemporaryFile
from contextlib import contextmanager
from pathlib import Path
import os
import numpy as np

from random import shuffle
from time import time


@contextmanager
def DataFile(size=2**20, rng=None):
    size = int(size)
    rng = rng or np.random.default_rng()
    with NamedTemporaryFile(mode="wb") as fo:

        #
        arr = np.empty(size, dtype="u1")
        arr[:] = rng.integers(256, size=size)
        fo.write(arr)
        fo.flush()

        # Get O_DIRECT file descriptor.
        # fo.close()
        fd = os.open(fo.name, os.O_DIRECT, os.O_RDONLY)

        path = Path(fo.name)

        yield arr, path, fd


def base_test(do_shuffle=True, read_specs=None, datafile=None, do_assert=True):

    # FULL_FILE_SIZE
    datafile = datafile or DataFile
    with datafile() as (
        arr,
        file_path,
        fd,
    ):

        NUM_READERS = 16
        if read_specs is None:
            BLOCK_SIZE = 4096
            NUM_INDICES = None

            size = file_path.stat().st_size
            indices = list(range(0, size, BLOCK_SIZE))
            if do_shuffle:
                shuffle(indices)
            indices = indices[:NUM_INDICES]

            read_specs = ((idx, BLOCK_SIZE) for idx in indices)

        t0 = time()
        data = list(
            mdl.raio_read(
                ((fd, idx, block_size, None) for (idx, block_size) in read_specs),
                NUM_READERS,
            )
        )
        t1 = time()
        bytes_read = sum(len(x) for x in data)
        delay = t1 - t0

        read_arr = np.concatenate(
            list(np.frombuffer(x.memview, dtype="u1") for x, ref in data)
        )

        if do_assert:
            if do_shuffle:
                read_arr.sort()
                arr.sort()
            npt.assert_array_equal(read_arr, arr)
        return arr, read_arr


def test_random():
    base_test(do_shuffle=True)


def test_sequential():
    base_test(do_shuffle=False)


def test_read_past_eof():
    with pytest.raises(
        Exception,
        match=re.escape(
            "Failed to read the requested number of bytes (read 512 but requested between 513 and 1024) !"
        ),
    ):
        base_test(
            do_shuffle=False,
            read_specs=[(2**10 - 1, 2)],
            datafile=lambda: DataFile(size=2**10),
        )


def test_read_past_eof_2():
    for read_spec, arr_slice in [
        [(2**10 - 1, 2), slice(-2, None)],
        [(2**10, 1), slice(-1, None)],
        [(0, 3), slice(None, 3)],
    ]:
        arr, read_arr = base_test(
            do_shuffle=False,
            read_specs=[read_spec],
            datafile=lambda: DataFile(size=2**10 + 1),
            do_assert=False,
        )
        npt.assert_array_equal(arr.__getitem__(arr_slice), read_arr)


def test_negative_offset():

    for read_specs in [(-1, 3), (1, -3)]:
        with pytest.raises(OverflowError):
            arr, read_arr = base_test(
                do_shuffle=False,
                read_specs=[read_specs],
                datafile=lambda: DataFile(size=2**10),
                do_assert=False,
            )


def test_negative_nbytes():
    pass


def test_read_short():
    for posn, numel in [(0, 1), (2**10 - 1, 1), (2**9, 5)]:
        arr, read_arr = base_test(
            do_shuffle=False,
            read_specs=[(posn, numel)],
            datafile=lambda: DataFile(size=2**10),
            do_assert=False,
        )
        assert len(read_arr) == numel
        npt.assert_array_equal(arr[posn : posn + numel], read_arr)


def test_doc():
    from pyraio import raio_read, raio_open_ctx
    from tempfile import NamedTemporaryFile
    import numpy as np

    with NamedTemporaryFile(mode="wb") as fo:

        # Write some data to the file.
        N = int(1e6)
        rng = np.random.default_rng()
        dat = rng.integers(256, size=N)
        fo.write(dat)
        fo.flush()

        # Read the data, 700 bytes at a time
        K = 700
        offsets = rng.permutation(range(0, N, K))
        with raio_open_ctx(fo.name) as fd:
            dat = list(raio_read((fd, offset, K, None) for offset in offsets))


class MyRef:
    def __init__(self, k):
        self.k = k


def test_ref():

    NUM_READERS = 32

    with DataFile() as (
        arr,
        file_path,
        fd,
    ):
        t0 = time()
        data = list(
            mdl.raio_read(
                (
                    (fd, idx, block_size, MyRef(k))
                    for k, (idx, block_size) in enumerate(
                        (k, 4096) for k in range(0, 2**20, 4096)
                    )
                ),
                NUM_READERS,
            )
        )

        data.sort(key=lambda x: x[1].k)
        read_arr = np.concatenate(
            list(np.frombuffer(x.memview, dtype="u1") for x, ref in data)
        )

        npt.assert_array_equal(read_arr, arr)
