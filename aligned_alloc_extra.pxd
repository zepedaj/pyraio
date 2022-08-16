
cdef inline size_t floor(size_t ALIGN_BNDRY, size_t num_bytes):
    return ALIGN_BNDRY*<size_t>(num_bytes/ALIGN_BNDRY)

cdef inline size_t ceil(size_t ALIGN_BNDRY, size_t num_bytes):
    return floor(ALIGN_BNDRY, num_bytes) + ALIGN_BNDRY*<size_t>(num_bytes%ALIGN_BNDRY>0)
