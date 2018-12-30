import argparse
import math
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from R1codes.contrast_enhancement import contrast
from R1codes.filter import fltr as filter_img
from R1codes.grayscale import conv_gray
from R1codes.histogram_exp import expand_hist
from R1codes.threshold import threshold
from R1codes.utils import im_to_arr
from utils import dylate, erode, quantize


def estimate_radious(binary_img, center_point):
    radious_estimates = []

    h, w = binary_img.shape
    # look for starting points of region
    # we ignore cases when region starts with image border, since those inforamations might be irrelevant
    i = 0
    for x in range(round(center_point[1])):
        for y in range(h):
            if binary_img[y, x] < 5 and binary_img[y, x+1] > 250:
                radious_estimates.append(
                    math.sqrt((center_point[1] - x)**2 + (center_point[0] - y)**2))
                i += 1
        if i > 0.5*h:
            break  # 50% is enough

    i = 0
    for x in range(w-2, round(center_point[1]+1), -1):
        for y in range(h):
            if binary_img[y, x] > 250 and binary_img[y, x+1] < 5:
                radious_estimates.append(
                    math.sqrt((center_point[1] - x)**2 + (center_point[0] - y)**2))
                i += 1
        if i > 0.5*h:
            break  # 50% is enough

    radious_estimates = sorted(radious_estimates)
    # trimmed mean
    trimmed = radious_estimates[round(
        len(radious_estimates)*0.1):round(len(radious_estimates)*0.8)]

    R = np.average(trimmed)
    return R


def find_outer_radious(img_arr, center_point):
    cont = np.copy(img_arr)
    cont = contrast(cont, 2)
    grayscale, _ = conv_gray(cont)

    grayscale, palette = quantize(grayscale, 4)

    # threshold all values smaller than maximal
    grayscale[grayscale < palette[-2] + 2] = 0
    grayscale[grayscale > 0] = 255

    grayscale = erode(grayscale, 7)
    grayscale = dylate(grayscale, 7)

    R = estimate_radious(grayscale, center_point)
    return R
