import pyraio.batch_reader as mdl
from multiprocessing import Process
from time import sleep
from contextlib import contextmanager
from pathlib import Path
import os

from random import shuffle
from time import time
import climax as clx

from tempfile import NamedTemporaryFile, TemporaryDirectory
import numpy as np

import subprocess as subp

#####################
from jzf_datasets.vanilla_dataset.cython.vanilla_batch_reader import (
    BatchReader as VanillaBatchReader,
)

from jzf_history import load_history
from itertools import islice


def build_history_input_iter(read_count, randomize, history_root, direct):
    history = load_history(history_root)
    batch_reader = VanillaBatchReader(
        history,
        sequence_len=100,
        batch_size=20000,
        depth=32,
        randomize=randomize,
        direct=direct,
    )
    return batch_reader.get_read_input_iter()


@contextmanager
def get_input_iter(prefix, path, read_count, block_size, randomize, direct):

    if path is not None and path.is_dir():
        # History directory
        _input_iter_obj = build_history_input_iter(
            read_count, randomize, path, direct=direct
        )
        input_iter = islice(_input_iter_obj, read_count)
        yield input_iter

    else:
        # Binary file or temporary binary file.
        with (
            datafile(prefix=prefix, direct=direct)
            if path is None
            else as_rdonly(path, direct=direct)
        ) as (
            file_path,
            fd,
            _,
        ):
            size = file_path.stat().st_size
            indices = list(range(0, size, block_size))[:-1]
            if randomize:
                shuffle(indices)
            indices = indices[:read_count]
            input_iter = ((fd, idx) for idx in indices)

            yield input_iter


#############


@contextmanager
def datafile(size=2**30, prefix=None, direct=False):

    size = int(size)
    rng = np.random.default_rng()
    with TemporaryDirectory(prefix=prefix) as tmp_dir:

        # #
        print("Creating a temporary file...")
        tmp_filename = Path(tmp_dir) / "file.bin"
        with open(tmp_filename, "wb") as fo:
            arr = np.empty(size, dtype="u1")
            arr[:] = rng.integers(255, size=size)
            fo.write(arr)
            fo.flush()
        # Without this sleep, read is slow!
        sleep(40)
        print("Done creating file.")

        # Get O_DIRECT file descriptor.
        with as_rdonly(tmp_filename, direct) as out:
            yield out


@contextmanager
def as_rdonly(filename, direct=False):
    fd = os.open(str(filename), os.O_RDONLY | (os.O_DIRECT if direct else 0))

    yield Path(filename), fd, None


@clx.command(
    "When not using option `--no-clear-io-cache`, sudo permissions might be required. If so, call using \n\tsudo env PATH=$PATH HOME=$HOME python benchmark.py <options>"
)
@clx.argument(
    "--path",
    type=Path,
    default=None,
    help="""The path to use - can be a file or a jzfin_dataset history directory. If not provided, a temporary file of size 1 GiB is generated internally by default using `--prefix`.
Example paths: '/data/mirrored/finance/alpaca/minute_unadjusted_sip/', '/data/tmp/large_file_3.tmp'""",  # TODO: Example files are not portable.
)
@clx.argument(
    "--prefix",
    type=str,
    default=None,
    help="The prefix used when creating a temporary file to read from. Defaults to the standard prefix. Only valid when `--path` is `None`.",
)
@clx.argument(
    "--block-size",
    type=int,
    default=4096,
    help="The size of each read request in bytes. Ignored if a history source is used (the block size is set by the record data type and sequence size in that case.",
)
@clx.argument(
    "--batch-size",
    type=lambda x: int(float(x)),
    default=int(2e4),
    help="The number of samples to read before yielding from cython.",
)
@clx.argument(
    "--depth", type=int, default=32, help="The number of parallel IO requests."
)
@clx.argument(
    "--read-count",
    type=lambda x: int(float(x)),
    default=None,
    help="Do at most this many reads (read the full file, by default).",
)
@clx.argument(
    "--no-randomize",
    dest="randomize",
    default=True,
    action="store_false",
    help="Disable read position randomization and instead read sequentially.",
)
@clx.argument(
    "--no-clear-io-cache",
    action="store_false",
    dest="clear_io_cache",
    help="Do not clear io caches before running the benchmark.",
)
@clx.argument(
    "--direct", action="store_true", help="Whether to open files in O_DIRECT mode."
)
def test_speed(
    path,
    block_size,
    depth,
    read_count,
    batch_size,
    prefix,
    randomize,
    clear_io_cache,
    direct,
):

    if clear_io_cache:
        out = subp.run(
            ["tee", "/proc/sys/vm/drop_caches"], input=b"3", stdout=subp.DEVNULL
        )
        if out.returncode:
            raise Exception(str(out))

    with get_input_iter(
        prefix, path, read_count, block_size, randomize, direct
    ) as input_iter:

        t0 = time()
        data = list(
            x[1]
            for x in mdl.raio_batch_read(
                input_iter, block_size, batch_size, depth=depth, direct=direct
            )
        )
        t1 = time()
        bytes_read = sum(x.size for x in data)
        delay = t1 - t0
        num_ios = len(data) * batch_size
        print(
            f"Read {bytes_read} bytes ({num_ios} ios, {len(data)} batches) in {delay} seconds  -- {bytes_read/delay/1e6} MBs | {num_ios/delay} iops."
        )


if __name__ == "__main__":
    test_speed()
