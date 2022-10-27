from pyraio import event_managers as mdl
from .batch_reader import DataFile
import numpy as np
import numpy.testing as npt
import os


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
