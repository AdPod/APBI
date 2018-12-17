#!/usr/bin/python
from PIL import Image
import numpy as np
import sys
from .utils import bound, im_to_arr


def contrast(img_arr, factor):
    return bound(((img_arr/255 - 0.5)*factor + 0.5)*255)


def main(args):
    factor = 8
    img = im_to_arr(args[0])
    ce = contrast(img, factor)
    ce = Image.fromarray(ce.astype(np.uint8))
    ce.show()


if __name__ == "__main__":
    main(sys.argv[1:])
