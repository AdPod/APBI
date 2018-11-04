#!/usr/bin/python
import sys

import numpy as np
from PIL import Image

from utils import im_to_arr, print_hist_img, occur_tab
from histogram_mainp import lut_mainp


def equalize_hist(arr):
    lut = np.zeros(256)
    h, w = arr.shape
    occur = occur_tab(arr)
    d0 = occur[0]/h/w
    const = 255/(1 - d0)
    for i in range(255):
        lut[i] = (np.sum(occur[:i+1])/h/w - d0)*const
    return lut_mainp(arr, lut)


def main(args):
    img = im_to_arr(args[0])

    r_channel = img[:, :, 0]
    g_channel = img[:, :, 1]
    b_channel = img[:, :, 2]

    equalized = np.zeros_like(img)
    equalized[:, :, 0] = equalize_hist(r_channel)
    equalized[:, :, 1] = equalize_hist(g_channel)
    equalized[:, :, 2] = equalize_hist(b_channel)

    Image.fromarray(img.astype(np.uint8)).show(title='original image')
    Image.fromarray(equalized.astype(np.uint8)).show(
        title='histogram equalized')

    print_hist_img([img, equalized])


if __name__ == "__main__":
    main(sys.argv[1:])
