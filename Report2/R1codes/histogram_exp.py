#!/usr/bin/python
import sys

import numpy as np
from PIL import Image

from .utils import im_to_arr, print_hist_img


def expand_color(arr):
    flat = arr.flatten()
    val_min = min(flat)
    val_max = max(flat)
    const = 255/(val_max - val_min)
    arr = const * (arr - val_min)
    return arr


def expand_hist(img_arr):
    r_channel = img_arr[:, :, 0]
    g_channel = img_arr[:, :, 1]
    b_channel = img_arr[:, :, 2]

    expanded = np.zeros_like(img_arr)
    expanded[:, :, 0] = expand_color(r_channel)
    expanded[:, :, 1] = expand_color(g_channel)
    expanded[:, :, 2] = expand_color(b_channel)
    return expanded


def main(args):
    img = im_to_arr(args[0])

    expanded = expand_hist(img)

    Image.fromarray(img.astype(np.uint8)).show(title='original image')
    Image.fromarray(expanded.astype(np.uint8)).show(title='histogram expanded')

    print_hist_img([img, expanded])


if __name__ == "__main__":
    main(sys.argv[1:])
