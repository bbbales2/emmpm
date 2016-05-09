#%%

import os
import numpy
import matplotlib.pyplot as plt
import skimage.filters
import skimage.io
import skimage.transform

os.chdir('/home/bbales2/emmpm')

image = skimage.io.imread('molybdenum12.png', as_grey = True) * 255.0#[:1200, :1200]

#image = skimage.transform.resize(image, (200, 200)) * 255.0
#%%

N = image.shape[0]

Y = image#numpy.random.randn(N, N) * 1.0
#Y[:N / 2, :] += 2.0
X = numpy.zeros(Y.shape).astype('int')
threshold = skimage.filters.threshold_otsu(Y)

X[Y > threshold] = 1

#X[:N / 2, :] = 1

b = 5.0
u = 0.1

samples = 10
rpts = 10

T = numpy.zeros((N, N, 2))
for (i, j), v in numpy.ndenumerate(X):
    T[i, j, v] = 1

for r in range(rpts):
    p0 = T[:, :, 0] / (T[:, :, 0] + T[:, :, 1])
    p1 = 1 - p0
    plt.imshow(p0)
    plt.show()

    #X = (p0 < 0.5).astype('int')

    u = {}
    sig2 = {}

    u[0] = sum((p0 * Y).flatten()) / sum(p0.flatten())
    u[1] = sum((p1 * Y).flatten()) / sum(p1.flatten())

    sig2[0] = sum((p0 * (Y - u[0])**2).flatten()) / sum(p0.flatten())
    sig2[1] = sum((p1 * (Y - u[1])**2).flatten()) / sum(p1.flatten())

    print u, [(k, numpy.sqrt(v)) for k, v in sig2.items()]

    T = numpy.zeros((N, N, 2))

    for s in range(samples):
        idxs = range(N * N)
        numpy.random.shuffle(idxs)

        ijs = numpy.unravel_index(idxs, (N, N))

        for i, j in zip(*ijs):
            im = max(i - 1, 0)
            ip = min(i + 1, N - 1)

            jm = max(j - 1, 0)
            jp = min(j + 1, N - 1)

            counts = { 0 : 0, 1 : 0 }

            counts[X[ip, j]] += 1
            counts[X[im, j]] += 1
            counts[X[i, jp]] += 1
            counts[X[i, jm]] += 1

            p0 = numpy.exp(-((Y[i, j] - u[X[i, j]])**2) / (2.0 * sig2[X[i, j]]) - b * counts[1 - X[i, j]])

            #print ((Y[i, j] - u[X[i, j]])**2) / (2.0 * sig2[X[i, j]])
            #print b * counts[1 - X[i, j]]
            p1 = numpy.exp(-((Y[i, j] - u[1 - X[i, j]])**2) / (2.0 * sig2[1 - X[i, j]]) - b * counts[X[i, j]])

            p = p0 / (p0 + p1)

            if numpy.random.random() > p:
                X[i, j] = 1 - X[i, j]

            T[i, j, X[i, j]] += 1
