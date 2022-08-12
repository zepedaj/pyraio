import pyraio as mdl
from pathlib import Path
import os

from random import shuffle
from time import time


def test_all():
    print("here")
    mdl.read_blocks([0])


def test_speed():

    # FULL_FILE_SIZE
    FILE_PATH = Path("/data/tmp/large_file_3.tmp")
    BLOCK_SIZE = 4096
    NUM_INDICES = 64000
    NUM_READERS = 32

    size = FILE_PATH.stat().st_size
    indices = list(range(0, size, BLOCK_SIZE))
    shuffle(indices)
    indices = indices[:NUM_INDICES]

    fd = os.open(FILE_PATH, os.O_RDONLY)  # | os.O_DIRECT)

    t0 = time()
    data = list(
        mdl.read_blocks(((fd, idx, BLOCK_SIZE) for idx in indices), NUM_READERS)
    )
    t1 = time()
    bytes_read = sum(len(x) for x in data)
    delay = t1 - t0
    print(
        f"Read {bytes_read} bytes ({len(data)} ios) in {delay} seconds  -- {bytes_read/delay/1e6} MBs | {len(data)/delay} iops."
    )


if __name__ == "__main__":
    main()

    pass
