import pyraio as mdl
from contextlib import contextmanager
from pathlib import Path
import os

from random import shuffle
from time import time
import climax as clx

from tempfile import NamedTemporaryFile
import numpy as np


@contextmanager
def datafile(size=2**30, rng=None):

    size = int(size)
    rng = rng or np.random.default_rng()
    with NamedTemporaryFile(mode="wb") as fo:

        #
        arr = np.empty(size, dtype="u1")
        arr[:] = rng.integers(255, size=size)
        fo.write(arr)

        # Get O_DIRECT file descriptor.
        # fo.close()
        fd = os.open(fo.name, os.O_DIRECT, os.O_RDONLY)

        yield Path(fo.name), fd, arr


@contextmanager
def as_o_direct_rdonly(filename):
    fd = os.open(str(filename), os.O_DIRECT | os.O_RDONLY)

    yield Path(filename), fd, None


@clx.command()
@clx.argument(
    "--filename",
    type=Path,
    default=None,
    help="The filename to use.A temporary file of size 1 GiB is generated internally by default.",
)
@clx.argument(
    "--block_size",
    type=int,
    default=4096,
    help="The size of each read request in bytes.",
)
@clx.argument(
    "--depth", type=int, default=32, help="The number of parallel IO requests."
)
@clx.argument(
    "--read_count",
    type=int,
    default=None,
    help="Do at most this many reads (read the full file, by default).",
)
def test_speed(filename, block_size, depth, read_count):

    with (datafile() if filename is None else as_o_direct_rdonly(filename)) as (
        file_path,
        fd,
        _,
    ):
        size = file_path.stat().st_size
        indices = list(range(0, size, block_size))
        shuffle(indices)
        indices = indices[:read_count]

        # fd = os.open(file_path, os.O_RDONLY)
        fd = os.open(file_path, os.O_RDONLY | os.O_DIRECT)

        t0 = time()
        data = list(
            mdl.raio_read(
                (
                    (
                        fd,
                        idx,
                        size - idx if (idx + block_size) > size else block_size,
                    )
                    for idx in indices
                ),
                depth,
            )
        )
        t1 = time()
        bytes_read = sum(len(x) for x in data)
        delay = t1 - t0
        print(
            f"Read {bytes_read} bytes ({len(data)} ios) in {delay} seconds  -- {bytes_read/delay/1e6} MBs | {len(data)/delay} iops."
        )


if __name__ == "__main__":
    test_speed()
