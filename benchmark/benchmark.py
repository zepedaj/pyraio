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


@contextmanager
def datafile(size=2**30, prefix=None):

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
        with as_o_direct_rdonly(tmp_filename) as out:
            yield out


@contextmanager
def as_o_direct_rdonly(filename):
    fd = os.open(str(filename), os.O_DIRECT | os.O_RDONLY)

    yield Path(filename), fd, None


@clx.command()
@clx.argument(
    "--filename",
    type=Path,
    default=None,
    help="The filename to use.A temporary file of size 1 GiB is generated internally by default using `--prefix`.",
)
@clx.argument(
    "--prefix",
    type=str,
    default=None,
    help="The prefix used when creating a temporary file to read from. Defaults to the standard prefix.",
)
@clx.argument(
    "--block-size",
    type=int,
    default=4096,
    help="The size of each read request in bytes.",
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
    type=int,
    default=None,
    help="Do at most this many reads (read the full file, by default).",
)
def test_speed(filename, block_size, depth, read_count, batch_size, prefix):

    with (
        datafile(prefix=prefix) if filename is None else as_o_direct_rdonly(filename)
    ) as (
        file_path,
        fd,
        _,
    ):
        size = file_path.stat().st_size
        indices = list(range(0, size, block_size))[:-1]
        shuffle(indices)
        indices = indices[:read_count]

        # fd = os.open(file_path, os.O_RDONLY)
        fd = os.open(file_path, os.O_RDONLY | os.O_DIRECT)

        t0 = time()
        data = list(
            x[1]
            for x in mdl.raio_batch_read(
                ((fd, idx) for idx in indices),
                block_size,
                batch_size,
                depth=depth,
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
