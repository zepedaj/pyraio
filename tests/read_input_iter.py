from pyraio import read_input_iter as mdl
import pytest
from itertools import chain
import os
from .batch_reader import DataFile


class TestReadInputIterWrapper:
    def test_all(self):
        num_batches = 10
        batch_size = 1000
        block_size = 100
        default_ref = -1
        with DataFile(
            size=(file_size := num_batches * batch_size * block_size), flags=os.O_RDONLY
        ) as (arr, path, fd):
            input_iter = [
                (fd, posn, default_ref) for posn in range(0, file_size, block_size)
            ]

            riiw = mdl.ReadInputIterWrapper(input_iter, default_ref)
            wrapped_input_iter_results = list(riiw)

            assert input_iter == wrapped_input_iter_results

    def test_next_exception(self):
        for source_iter in [[(-1, 0)], [(0, -1)]]:
            with pytest.raises(OverflowError):
                list(mdl.ReadInputIterWrapper(source_iter))


class TestReadInputIterChunk:
    def test_all(self):
        num_batches = 10
        batch_size = 1000
        block_size = 100
        default_ref = -1
        with DataFile(
            size=(file_size := num_batches * batch_size * block_size), flags=os.O_RDONLY
        ) as (arr, path, fd):
            input_iter = [
                (fd, posn, default_ref) for posn in range(0, file_size, block_size)
            ]

            rii = mdl.ReadInputIterWrapper(input_iter, default_ref)
            vals_from_chunks = list(chain(*list(rii.iter_chunks(2))))

            rii = mdl.ReadInputIterWrapper(input_iter, default_ref)
            vals = list(rii)

            assert len(vals) == len(input_iter)
            assert sorted(vals) == sorted(vals_from_chunks)

    def test_next_exception(self):
        for source_iter in [[(-1, 0)], [(0, -1)]]:
            rii = mdl.ReadInputIterWrapper(source_iter)
            with pytest.raises(OverflowError):
                list(rii.iter_chunks(2))
