from skimage import io
from skimage.color import rgb2gray
import random
from random import randint
from math import ceil, floor
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
from utils import synthRandomPatch, synthEfrosLeung

# Load images
#img = rgb2gray(io.imread('../data/texture/D20.png')[:,:,:3])
img = rgb2gray(io.imread('../hw4/data/texture/Texture2.bmp')[:,:,:3])
#img = rgb2gray(io.imread('../data/texture/english.jpg')[:,:,:3])

# Random patches
tileSize = 30 # specify block sizes
numTiles = 5
outSize = numTiles * tileSize # calculate output image size

# # implement the following, save the random-patch output and record run-times
# im_patch = synthRandomPatch(img, tileSize, numTiles, outSize)
# plt.imshow(im_patch, cmap='gray')
# plt.show()
# plt.savefig('random_patch.png')


# Non-parametric Texture Synthesis using Efros & Leung algorithm  
winsize = 11 # specify window size (5, 7, 11, 15)
outSize = 70 # specify size of the output image to be synthesized (square for simplicity)
# implement the following, save the synthesized image and record the run-times
im_synth = synthEfrosLeung(img, winsize, outSize)
plt.imshow(im_synth, cmap='gray')
plt.show()
plt.savefig('synth_efros_leung.png')
