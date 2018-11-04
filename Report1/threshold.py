#!/usr/bin/python
import sys

import numpy as np
from PIL import Image

from grayscale import conv_gray
from utils import bound, im_to_arr


def threshold(arr, t=80):
    t = max(0, min(t, 255))
    wei, avg = conv_gray(arr)
    thresholded = np.zeros_like(wei)
    thresholded[wei > t] = 255
    return thresholded


def main(args):

    img = im_to_arr(args[0])

    thresholded = threshold(img)

    thresholded = Image.fromarray(thresholded.astype(np.uint8))
    thresholded.show()


if __name__ == "__main__":
    main(sys.argv[1:])
