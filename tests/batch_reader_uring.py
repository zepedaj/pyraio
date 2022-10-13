from pyraio import batch_reader_uring as mdl
import os
import numpy as np
import numpy.testing as npt
from .batch_reader import DataFile


class TestBlockManager:
    def test_all(self):
        bm = mdl.BlockManager(100, 32)

    def test_read(self):
        block_size = int(1e2)
        batch_size = 1000
        depth = 32
        next_submission = 0
        retrieved = []

        bm = mdl.BlockManager(block_size, depth)

        with DataFile(size=block_size * batch_size, flags=os.O_RDONLY) as (
            arr,
            path,
            fd,
        ):

            out = np.zeros_like(arr).reshape(batch_size, block_size)

            while next_submission < batch_size:
                if bm._num_pending == depth:
                    while bm._num_pending > 0:

                        retrieved.append(bm._get_completed(True))
                bm._submit(
                    fd,
                    out[next_submission].ctypes.data,
                    block_size,
                    next_submission * block_size,
                    next_submission,
                )
                next_submission += 1

            while bm._num_pending:
                retrieved.append(bm._get_completed(True))

            assert sorted(retrieved) == list(range(batch_size))

            npt.assert_array_equal(out.reshape(-1), arr)

    def test_perr_enum(self):
        x = mdl.PERR
        assert min(x) == mdl.PERR_START
        assert max(x) >= mdl.PERR_START
