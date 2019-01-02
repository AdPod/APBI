import argparse
import math
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from inner_circle import find_inner_circle
from outer_circle import find_outer_radius
from R1codes.contrast_enhancement import contrast
from R1codes.filter import fltr as filter_img
from R1codes.grayscale import conv_gray
from R1codes.histogram_exp import expand_hist
from R1codes.threshold import threshold
from R1codes.utils import im_to_arr
from utils import bw_padding, dilate, erode


def crop(R, r, starting_point, img_path):
    diameter = 2*R
    scalled_mask_size = (diameter*3, diameter*3)
    mask = Image.new('L', scalled_mask_size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0)+scalled_mask_size, fill=255)
    draw.ellipse((3*(R-r), 3*(R-r), 3*(R+r), 3*(R+r)), fill=0)
    mask = mask.resize((diameter, diameter), Image.ANTIALIAS)

    img = Image.open(img_path)

    h_diff = (mask.size[1] - img.size[1])/2
    w_diff = (mask.size[0] - img.size[0])/2
    mask = mask.crop(
        (w_diff, h_diff, mask.size[0] - w_diff, mask.size[1] - h_diff))
    mask = mask.resize(img.size, Image.ANTIALIAS)
    output = ImageOps.fit(img, mask.size, centering=(0.5, 0.5))
    output.putalpha(mask)
    output.save('./out-transparent.png')

    # below fix for .show(), which fails to show transparency
    bg = Image.new('RGB', output.size, (0, 0, 0))
    bg.paste(output, output)
    bg.show()
    bg.save('./out.png')


def main(args):
    img = im_to_arr(args.img)

    # prepare image
    img = filter_img(img, 'gauss')
    img = filter_img(img, 'gauss')
    img = expand_hist(img)

    r, center = find_inner_circle(img)
    R = find_outer_radius(img, center)

    crop(int(round(R)), int(round(r)), center, args.img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="iris segmentation")
    parser.add_argument('-i', '--img', required=True)
    args = parser.parse_args()
    main(args)
