#%%

import os
import numpy
import matplotlib.pyplot as plt
import skimage.filters
import skimage.io
import skimage.transform

os.chdir('/home/bbales2/emmpm')

import pyximport
pyximport.install()

import emmpm

image = skimage.io.imread('molybdenum12.png', as_grey = True) * 255.0

import time

tmp = time.time()
out = emmpm.segment(image)
plt.imshow(out)
print time.time() - tmp
plt.colorbar()
plt.show()
