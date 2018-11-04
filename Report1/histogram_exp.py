#!/usr/bin/python
import sys

import numpy as np
from PIL import Image

from utils import im_to_arr, print_hist_img


def expand_hist(arr):
    flat = arr.flatten()
    val_min = min(flat)
    val_max = max(flat)
    const = 255/(val_max - val_min)
    arr = const * (arr - val_min)
    return arr


def main(args):
    img = im_to_arr(args[0])

    r_channel = img[:, :, 0]
    g_channel = img[:, :, 1]
    b_channel = img[:, :, 2]
    
    expanded = np.zeros_like(img)
    expanded[:, :, 0] = expand_hist(r_channel)
    expanded[:, :, 1] = expand_hist(g_channel)
    expanded[:, :, 2] = expand_hist(b_channel)


    Image.fromarray(img.astype(np.uint8)).show(title='original image')
    Image.fromarray(expanded.astype(np.uint8)).show(title='histogram expanded')

    print_hist_img([img, expanded])

if __name__ == "__main__":
    main(sys.argv[1:])
