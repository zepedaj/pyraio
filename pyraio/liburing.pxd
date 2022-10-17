
cdef extern from "<liburing.h>" nogil:

    ctypedef struct io_uring:
        int ring_fd

    ctypedef unsigned __u64 # Cython replaces the correct type in the header file. See https://cython.readthedocs.io/en/latest/src/userguide/external_C_code.html
    ctypedef unsigned __u8
    ctypedef signed __s64
    ctypedef signed __s32

    int EAGAIN
    int IORING_REGISTER_FILES
    int IOSQE_FIXED_FILE

    int io_uring_queue_init(unsigned entries, io_uring *ring, unsigned flags)
    void io_uring_queue_exit(io_uring *ring)
    io_uring_sqe *io_uring_get_sqe(io_uring *ring)
    void io_uring_prep_read(io_uring_sqe *sqe, int fd, void *buf, unsigned nbytes, __u64 offset)
    int io_uring_submit(io_uring *ring)
    int io_uring_peek_cqe(io_uring *ring, io_uring_cqe **cqe_ptr)
    int io_uring_wait_cqe(io_uring *ring, io_uring_cqe **cqe_ptr)
    void io_uring_cqe_seen(io_uring *ring, io_uring_cqe *cqe)
    int io_uring_register(unsigned int fd, unsigned int opcode, void *arg, unsigned int nr_args)

    cdef extern from "<liburing/io_uring.h>" nogil:

        ctypedef struct io_uring_sqe:
            __u64 user_data
            __u8 flags
            __s32 fd

        ctypedef struct io_uring_cqe:
            __u64 user_data
            __s32 res
