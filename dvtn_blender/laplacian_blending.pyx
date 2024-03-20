# cython: language_level=3

import numpy as np
cimport numpy as cnp

from libc.stdint cimport int32_t
from cpython cimport array
import array

# Import OpenCV Python bindings
import cv2

def laplacian_blending(cnp.ndarray[cnp.float32_t, ndim=3] A, cnp.ndarray[cnp.float32_t, ndim=3] B, cnp.ndarray[cnp.float32_t, ndim=2] m, int num_levels=7):
    cdef int height = m.shape[0]
    cdef int width = m.shape[1]
    cdef cnp.ndarray[cnp.int32_t, ndim=1] size_list = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], dtype=np.int32)
    cdef int size = size_list[np.where(size_list > max(height, width))][0]
    cdef cnp.ndarray[cnp.float32_t, ndim=3] GA = np.zeros((size, size, 3), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=3] GB = np.zeros((size, size, 3), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=3] GM = np.zeros((size, size, 3), dtype=np.float32)
    cdef list gpA = [GA]
    cdef list gpB = [GB]
    cdef list gpM = [GM]
    cdef int i
    for i in range(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))
    cdef list lpA = [gpA[num_levels-1]]
    cdef list lpB = [gpB[num_levels-1]]
    cdef list gpMr = [gpM[num_levels-1]]
    for i in range(num_levels-1, 0, -1):
        LA = np.subtract(gpA[i-1], cv2.pyrUp(gpA[i]))
        LB = np.subtract(gpB[i-1], cv2.pyrUp(gpB[i]))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i-1])
    cdef list LS = []
    for la, lb, gm in zip(lpA, lpB, gpMr):
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)
    cdef cnp.ndarray[cnp.float32_t, ndim=3] ls_ = LS[0]
    for i in range(1, num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])
    ls_ = ls_[:height, :width, :]
    return ls_.clip(0, 255)
