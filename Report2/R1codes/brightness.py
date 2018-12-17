#!/usr/bin/python
from PIL import Image
import numpy as np
import sys
from utils import bound, im_to_arr

def main(args):
    m = 100
    img = im_to_arr(args[0])
    m = max(0, min(m, 255))
    inv = bound(img + m)
    inv = Image.fromarray(inv.astype(np.uint8))
    inv.show()


if __name__ == "__main__":
    main(sys.argv[1:])
