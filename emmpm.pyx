#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy
cimport numpy
cimport libc.math
import matplotlib.pyplot as plt
import skimage.filters

cpdef segment(image, double b = 5.0):
    cdef int N = image.shape[0]

    cdef numpy.ndarray[numpy.double_t, ndim = 2] Y = image.astype('double')
    cdef numpy.ndarray[numpy.uint64_t, ndim = 2] X = numpy.zeros((N, N)).astype('uint64')

    cdef numpy.ndarray[numpy.double_t, ndim = 2] p = numpy.zeros((N, N))
    
    threshold = skimage.filters.threshold_otsu(Y)

    X[Y > threshold] = 1

    cdef int samples = 10
    cdef int rpts = 10

    cdef numpy.ndarray[numpy.uint64_t, ndim = 3] T = numpy.zeros((N, N, 2)).astype('uint64')
    cdef numpy.ndarray[numpy.uint64_t, ndim = 2] TT = numpy.zeros((N, N)).astype('uint64')

    cdef int i, j, v

    for i in range(N):
        for j in range(N):
            T[i, j, X[i, j]] = 1

    cdef int r, tmp
    cdef double[2] u = [0.0, 0.0]
    cdef double[2] sig2 = [0.0, 0.0]
    cdef double psum
    cdef int[2] counts
    cdef double p0, p1

    cdef numpy.ndarray[numpy.double_t, ndim = 2] randoms

    for r in range(rpts):
        for i in range(N):
            for j in range(N):
                tmp = 0

                for c in range(2):
                    tmp += T[i, j, c]

                TT[i, j] = tmp

        for c in range(2):
            for i in range(N):
                for j in range(N):
                    p[i, j] = T[i, j, c] / TT[i, j]

            psum = 0.0

            for i in range(N):
                for j in range(N):
                    psum += p[i, j]
                    u[c] += p[i, j] * Y[i, j]

            u[c] /= psum

            for i in range(N):
                for j in range(N):
                    sig2[c] += p[i, j] * (Y[i, j] - u[c])**2

            sig2[c] /= psum

        print u, sig2

        T = numpy.zeros((N, N, 2)).astype('uint64')

        for s in range(samples):
            # The paper "The EM/MPM Algorithm for Segmentation of Textured Images: Analysis and Further Experimental Results" says I should only change one pixel at a time
            #   I saw an implementation from http://www.bluequartz.net/ (EMMPM workbench) where I think
            #   they changed more than one pixel at a time (I don't think I really understood the code so I could be wrong).
            #
            #   I think changing more than one pixel is fine. You're just making bigger jumps in state space,
            #   so I'm doin' it here too.
            #
            for i in range(N):
                for j in range(N):
                    im = 0 if i - 1 < 0 else i - 1
                    ip = i + 1 if i + 1 < N else N - 1

                    jm = 0 if j - 1 < 0 else j - 1
                    jp = j + 1 if j + 1 < N else N - 1

                    counts = [0, 0]

                    counts[X[ip, j]] += 1
                    counts[X[im, j]] += 1
                    counts[X[i, jp]] += 1
                    counts[X[i, jm]] += 1

                    p0 = libc.math.exp(-((Y[i, j] - u[X[i, j]])**2) / (2.0 * sig2[X[i, j]]) - b * counts[1 - X[i, j]])
                    p1 = libc.math.exp(-((Y[i, j] - u[1 - X[i, j]])**2) / (2.0 * sig2[1 - X[i, j]]) - b * counts[X[i, j]])

                    p[i, j] = p0 / (p0 + p1)

            randoms = numpy.random.rand(N, N)

            for i in range(N):
                for j in range(N):
                    if randoms[i, j] > p[i, j]:
                        X[i, j] = 1 - X[i, j]

                    T[i, j, X[i, j]] += 1

    return T[:, :, 1] / (T[:, :, 0] + T[:, :, 1]).astype('double')

