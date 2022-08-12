cdef extern from "<libaio.h>" nogil:

    ctypedef struct io_context:
        pass

    ctypedef struct sigset_t:
        pass

    cdef struct timespec

    cdef struct sockaddr

    cdef struct iovec

    ctypedef io_context* io_context_t

    cpdef enum io_iocb_cmd:
        IO_CMD_PREAD
        IO_CMD_PWRITE
        IO_CMD_FSYNC
        IO_CMD_FDSYNC
        IO_CMD_POLL
        IO_CMD_NOOP
        IO_CMD_PREADV
        IO_CMD_PWRITEV

    ctypedef io_iocb_cmd io_iocb_cmd_t

    cdef struct io_iocb_poll:
        int events
        int __pad1

    cdef struct io_iocb_sockaddr:
        sockaddr* addr
        int len

    cdef struct io_iocb_common:
        void* buf
        unsigned long nbytes
        long long offset
        long long __pad3
        unsigned flags
        unsigned resfd

    cdef struct io_iocb_vector:
        iovec* vec
        int nr
        long long offset

    cdef union _iocb_u_u:
        io_iocb_common c
        io_iocb_vector v
        io_iocb_poll poll
        io_iocb_sockaddr saddr

    cdef struct iocb:
        void* data
        unsigned key
        unsigned aio_rw_flags
        short aio_lio_opcode
        short aio_reqprio
        int aio_fildes
        _iocb_u_u u

    cdef struct io_event:
        void* data
        iocb* obj
        unsigned long res
        unsigned long res2

    cdef struct io_sigset:
        unsigned long ss
        unsigned long ss_len

    cdef struct io_sigset_compat:
        unsigned long ss
        unsigned long ss_len

    ctypedef void (*io_callback_t)(io_context_t ctx, iocb* iocb, long res, long res2)

    int io_queue_init(int maxevents, io_context_t* ctxp)

    int io_queue_release(io_context_t ctx)

    int io_queue_run(io_context_t ctx)

    int io_setup(int maxevents, io_context_t* ctxp)

    int io_destroy(io_context_t ctx)

    int io_submit(io_context_t ctx, long nr, iocb* ios[])

    int io_cancel(io_context_t ctx, iocb* iocb, io_event* evt)

    int io_getevents(io_context_t ctx, long min_nr, long nr, io_event* events, timespec* timeout)

    int io_pgetevents(io_context_t ctx, long min_nr, long nr, io_event* events, timespec* timeout, sigset_t* sigmask)

    void io_set_callback(iocb* iocb, io_callback_t cb)

    void io_prep_pread(iocb* iocb, int fd, void* buf, size_t count, long long offset)

    void io_prep_pwrite(iocb* iocb, int fd, void* buf, size_t count, long long offset)

    void io_prep_preadv(iocb* iocb, int fd, iovec* iov, int iovcnt, long long offset)

    void io_prep_pwritev(iocb* iocb, int fd, iovec* iov, int iovcnt, long long offset)

    void io_prep_preadv2(iocb* iocb, int fd, iovec* iov, int iovcnt, long long offset, int flags)

    void io_prep_pwritev2(iocb* iocb, int fd, iovec* iov, int iovcnt, long long offset, int flags)

    void io_prep_poll(iocb* iocb, int fd, int events)

    int io_poll(io_context_t ctx, iocb* iocb, io_callback_t cb, int fd, int events)

    void io_prep_fsync(iocb* iocb, int fd)

    int io_fsync(io_context_t ctx, iocb* iocb, io_callback_t cb, int fd)

    void io_prep_fdsync(iocb* iocb, int fd)

    int io_fdsync(io_context_t ctx, iocb* iocb, io_callback_t cb, int fd)

    void io_set_eventfd(iocb* iocb, int eventfd)
