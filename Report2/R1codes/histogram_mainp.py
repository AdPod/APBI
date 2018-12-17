#!/usr/bin/python
import sys

import numpy as np
from PIL import Image

from utils import bound, im_to_arr


def lut_mainp(img, lut):
    w, h = img.shape
    flat = img.flatten()
    for x in range(len(flat)):
        flat[x] = lut[flat[x]]
    flat = flat.reshape((w, h))
    return flat


def main(args):
    img = im_to_arr(args[0])

    r_lut = bound(np.array(range(0, 256)) + 50)
    g_lut = bound(np.array(range(0, 256)) + 40)
    b_lut = bound(np.array(range(0, 256)) - 100)
    img[:, :, 0] = lut_mainp(img[:, :, 2], r_lut)
    img[:, :, 1] = lut_mainp(img[:, :, 2], g_lut)
    img[:, :, 2] = lut_mainp(img[:, :, 2], b_lut)

    img = Image.fromarray(img.astype(np.uint8))
    img.show()


if __name__ == "__main__":
    main(sys.argv[1:])
