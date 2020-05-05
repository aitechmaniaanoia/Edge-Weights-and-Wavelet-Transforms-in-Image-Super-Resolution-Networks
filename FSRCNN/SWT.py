# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 19:41:59 2020

@author: Zoe
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt
#import pywt.data
# Load image
import skimage
from skimage import data
from skimage.color import rgb2gray
from skimage import io
import cv2
#img = io.imread(path_)

#original = data.astronaut()
original = io.imread('0001.png')
original = rgb2gray(original)


# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(original, 'haar')
#coeffs2 = pywt.swt2(original, 'db1', level = 1)
LL, (LH, HL, HH) = coeffs2

cv2.imwrite('h.jpg',LH*255)
cv2.imwrite('v.jpg',HL*255)
cv2.imwrite('d.jpg',HH*255)

mask = LH + HL + HH
mask[mask < np.mean(mask)] = 0
mask[mask > 0] = 1

cv2.imwrite('mask1.jpg',mask*255)


fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    #ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.imshow(abs(a), cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    
fig.tight_layout()
plt.show()

#cv2.imwrite('aa.jpg',fig)