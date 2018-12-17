#!/usr/bin/python
import sys

import numpy as np
from PIL import Image

from .utils import im_to_arr


def conv_gray(arr):
    (w, h) = arr.shape[:2]
    avg = np.zeros((w, h), np.int16)
    wei = np.zeros((w, h), np.int16)
    sh = arr.shape
    for x in range(w):
        for y in range(h):
            r, g, b = arr[x, y]
            avg[x, y] = (r + g + b)/3
            wei[x, y] = 0.299*r + 0.587*g + 0.144*b
    return wei, avg


def main(args):
    img = im_to_arr(args[0])
    wei, avg = conv_gray(img)
    wei = Image.fromarray(wei)
    avg = Image.fromarray(avg)
    wei.show()
    avg.show()
    # wei.save('wieghted_grayscale.png')
    # avg.save('average_grayscale.png')


if __name__ == "__main__":
    main(sys.argv[1:])
