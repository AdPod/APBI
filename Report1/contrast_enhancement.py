#!/usr/bin/python
from PIL import Image
import numpy as np
import sys
from utils import bound, im_to_arr


def main(args):
    factor = 8
    img = im_to_arr(args[0])
    ce = bound(((img/255 - 0.5)*factor + 0.5)*255)
    ce = Image.fromarray(ce.astype(np.uint8))
    ce.show()


if __name__ == "__main__":
    main(sys.argv[1:])
