from pyraio import batch_reader as mdl
import os
import numpy as np
import numpy.testing as npt

from sys import getrefcount
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
def DataFile(size=2**20, rng=None, flags=os.O_RDONLY):
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
        fd = os.open(fo.name, flags)

        path = Path(fo.name)

        yield arr, path, fd


class TestEventManager:
    cls = lambda self, block_size, depth: mdl.EventManager(depth=depth)
    flags = os.O_RDONLY

    def test_all(self):
        bm = self.cls(100, 32)

    def test_read(self):
        block_size = int(1e2)
        batch_size = 10000
        depth = 32
        next_submission = 0
        retrieved = []

        bm = self.cls(block_size, depth)

        with DataFile(size=block_size * batch_size, flags=self.flags) as (
            arr,
            path,
            fd,
        ):

            out = np.zeros_like(arr).reshape(batch_size, block_size)

            while next_submission < batch_size:
                if bm._num_submitted == depth:
                    while bm._num_submitted > 0:
                        retrieved.append(bm._get_completed(True))
                bm._enqueue(
                    fd,
                    out[next_submission].ctypes.data,
                    block_size,
                    next_submission * block_size,
                )
                bm._submit()
                next_submission += 1

            while bm._num_submitted:
                retrieved.append(bm._get_completed(True))

            npt.assert_array_equal(out.reshape(-1), arr)

    def test_perr_enum(self):
        x = mdl.PERR
        assert min(x) == mdl.PERR_START
        assert max(x) >= mdl.PERR_START


class TestDirectEventManager(TestEventManager):
    cls = lambda self, block_size, depth: mdl.DirectEventManager(
        max_block_size=block_size, depth=depth
    )
    flags = os.O_RDONLY | os.O_DIRECT


class MyRef:
    def __init__(self, k):
        self.k = k


class TestRAIOBatchReader:
    raio_batch_read = mdl.raio_batch_read

    def base_raio_batch_read(
        self, datafile=None, indices=None, block_size=4096, batch_size=1
    ):
        # FULL_FILE_SIZE
        datafile = datafile or DataFile
        with datafile() as (
            arr,
            file_path,
            fd,
        ):

            NUM_READERS = 16
            if indices is None:
                NUM_INDICES = None

                size = file_path.stat().st_size
                indices = list(range(0, size, block_size))
                shuffle(indices)
                indices = indices[:NUM_INDICES]

            data = list(
                self.raio_batch_read(
                    ((fd, idx) for idx in indices),
                    block_size,
                    batch_size,
                    NUM_READERS,
                )
            )

            return arr, data

    def base_test(
        self,
        do_shuffle=True,
        indices=None,
        block_size=4096,
        datafile=None,
        do_assert=True,
        batch_size=1,
    ):

        # FULL_FILE_SIZE
        datafile = datafile or DataFile
        with datafile() as (
            arr,
            file_path,
            fd,
        ):

            NUM_READERS = 16
            if indices is None:
                NUM_INDICES = None

                size = file_path.stat().st_size
                indices = list(range(0, size, block_size))
                if do_shuffle:
                    shuffle(indices)
                indices = indices[:NUM_INDICES]

            t0 = time()
            batches = list(
                self.raio_batch_read(
                    ((fd, idx) for idx in indices),
                    block_size,
                    batch_size,
                    NUM_READERS,
                )
            )

            t1 = time()
            bytes_read = sum(x[1].size for x in batches)
            delay = t1 - t0

            read_arr = np.concatenate(
                list(np.frombuffer(data, dtype="u1") for ref, data in batches)
            )

            if do_assert:
                if do_shuffle:
                    # TODO: No need to do this!
                    read_arr.sort()
                    arr.sort()
                npt.assert_array_equal(read_arr, arr)
            return arr, read_arr

    def test_random(self):
        self.base_test(do_shuffle=True)

    def test_sequential(self):
        self.base_test(do_shuffle=False)

    mismatched_bytes_expr = re.escape(
        "Failed to read the expected number of bytes: 1 byte(s) read, but expected 2!"
    )

    def test_read_past_eof(self):
        with pytest.raises(Exception, match=self.mismatched_bytes_expr):
            self.base_test(
                do_shuffle=False,
                indices=[2**10 - 1],
                block_size=2,
                datafile=lambda: DataFile(size=2**10),
            )

    def test_read_past_eof_2(self):
        for (indices, block_size), arr_slice in [
            [([2**10 - 1], 2), slice(-2, None)],
            [([2**10], 1), slice(-1, None)],
            [([0], 3), slice(None, 3)],
        ]:
            arr, read_arr = self.base_test(
                do_shuffle=False,
                indices=indices,
                block_size=block_size,
                datafile=lambda: DataFile(size=2**10 + 1),
                do_assert=False,
            )
            npt.assert_array_equal(arr.__getitem__(arr_slice), read_arr)

    def test_negative_offset(self):
        for indices, block_size in [([-1], 3), ([1], -3)]:
            with pytest.raises(OverflowError):
                arr, read_arr = self.base_test(
                    do_shuffle=False,
                    indices=indices,
                    block_size=block_size,
                    datafile=lambda: DataFile(size=2**10),
                    do_assert=False,
                )

    def test_negative_nbytes(self):
        pass

    def test_read_short(self):
        for posn, block_size in [(0, 1), (2**10 - 1, 1), (2**9, 5)]:
            arr, read_arr = self.base_test(
                do_shuffle=False,
                indices=[posn],
                block_size=block_size,
                datafile=lambda: DataFile(size=2**10),
                do_assert=False,
            )
            assert len(read_arr) == block_size
            npt.assert_array_equal(arr[posn : posn + block_size], read_arr)

    def test_doc(self):
        from pyraio import raio_batch_read
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

            with open(fo.name) as fo:
                fd = fo.fileno()
                dat = [
                    batch
                    for ref, batch in raio_batch_read(
                        ((fd, offset) for offset in offsets), num_bytes, batch_size
                    )
                ]

            assert [_x.size for _x in dat] == [batch_size * num_bytes] * num_batches

    def test_last_batch_pruned(self):
        #
        posns, block_size, batch_size = ([0, 2**10 - 400, 2**10 - 100], 100, 2)
        #
        arr, data = self.base_raio_batch_read(
            indices=posns,
            block_size=block_size,
            batch_size=batch_size,
            datafile=lambda: DataFile(size=2**10),
        )

        assert len(data) == np.ceil(len(posns) / batch_size)
        batches = [_x[1] for _x in data]
        batch_lens = [_x.size for _x in batches]
        expected_batch_lens = [block_size * batch_size] * (len(posns) // batch_size) + [
            (len(posns) % batch_size) * block_size
        ] * (int(bool(len(posns) % batch_size)))
        assert batch_lens == expected_batch_lens
        # TODO:
        #
        read_dat = np.concatenate(batches)
        actual_dat = np.concatenate(
            [arr[_posn : _posn + block_size][None, :] for _posn in posns]
        )
        actual_dat.sort()
        read_dat.sort()
        npt.assert_array_equal(actual_dat, read_dat)

    def test_ref(self):

        NUM_READERS = 32
        block_size = 4096

        with DataFile() as (
            arr,
            file_path,
            fd,
        ):
            t0 = time()
            data = list(
                self.raio_batch_read(
                    ((fd, idx, k) for k, idx in enumerate(range(0, 2**20, 4096))),
                    block_size,
                    1,
                    NUM_READERS,
                    ref_map=[MyRef(k) for k in range(2**20 // 4096)],
                )
            )

            data.sort(key=lambda x: x[0][0].k)
            for k in range(len(data)):
                _ref_list, _buf = data[k]
                # Ref count = 3 bc of original ref, _buf, and getrefcount param.
                assert getrefcount(_buf) == 3
                assert getrefcount(_ref_list) == 3
                assert isinstance(_ref_list, list)
                for l in range(len(_ref_list)):
                    _ref = _ref_list[l]
                    assert getrefcount(_ref) == 3
                    assert isinstance(_ref, MyRef)
            read_arr = np.concatenate(
                list(np.frombuffer(x, dtype="u1") for ref, x in data)
            )

            npt.assert_array_equal(read_arr, arr)


class TestRAIOBatchReaderDirect(TestRAIOBatchReader):
    raio_batch_read = lambda self, *args, **kwargs: mdl.raio_batch_read(
        *args, direct=True, **kwargs
    )

    mismatched_bytes_expr = re.escape(
        "Failed to read the expected number of bytes: 512 byte(s) read, but expected at least 513!"
    )
