import pyraio as mdl
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
            mdl.read_blocks(
                ((fd, idx, block_size) for (idx, block_size) in read_specs), NUM_READERS
            )
        )
        t1 = time()
        bytes_read = sum(len(x) for x in data)
        delay = t1 - t0

        read_arr = np.concatenate(
            list(np.frombuffer(x.memview, dtype="u1") for x in data)
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
        Exception, match="Failed to read the requested number of bytes!"
    ):
        base_test(
            do_shuffle=False,
            read_specs=[(2**10 - 1, 2)],
            datafile=lambda: DataFile(size=2**10),
        )


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
