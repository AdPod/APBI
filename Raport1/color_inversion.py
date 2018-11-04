#!/usr/bin/python
import sys

import numpy as np
from PIL import Image

from utils import bound


def main(args):
    try:
        img = Image.open(args[0])
    except FileNotFoundError:
        print('usege: python color_inversion.py <name of a file to convert>')
    else:
        img = np.array(img.convert('RGB'))

        inv = bound(255 - img)
        inv = Image.fromarray(inv).convert("RGB")
        inv.show()


if __name__ == "__main__":
    main(sys.argv[1:])
