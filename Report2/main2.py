import argparse
import math
import sys

import numpy as np
from PIL import Image, ImageDraw

from R1codes.contrast_enhancement import contrast
from R1codes.filter import fltr as filter_img
from R1codes.grayscale import conv_gray
from R1codes.histogram_exp import expand_hist
from R1codes.threshold import threshold
from R1codes.utils import im_to_arr


def bw_padding(arr, size):
    padded = np.zeros((arr.shape[0] + 2*size, arr.shape[1] + 2*size))
    padded[size:-size, size:-size] = arr
    # mirror padding
    for x in range(size):
        padded[x, :] = padded[size, :]
        padded[-1-x, :] = padded[-1-size, :]
    for y in range(size):
        padded[:, y] = padded[:, size]
        padded[:, -1-x] = padded[:, -1-size]

    return padded


def dylatation(img_arr):
    working = bw_padding(img_arr, 2)
    for x in range(img_arr.shape[0]):
        for y in range(img_arr.shape[1]):
            if (np.sum(working[x:x+5, y:y+5]) - working[x+2, y+2]) > 0:
                img_arr[x, y] = 255
    return img_arr


def erosion(img_arr):
    working = bw_padding(img_arr, 2)
    for x in range(img_arr.shape[0]):
        for y in range(img_arr.shape[1]):
            if (np.sum(working[x:x+5, y:y+5]) - working[x+2, y+2]) < 6120:
                img_arr[x, y] = 0
    return img_arr


def main(args):
    img = im_to_arr(
        '/home/dominik/Stud/Biometria/Report2/images/eye_unsharp.png')
    Image.fromarray(img.astype(np.uint8)).show()
    cont = img
    cont = contrast(cont, 1.2)
    Image.fromarray(cont.astype(np.uint8)).show()
    grayscale, _ = conv_gray(cont)
    flatten = np.array(sorted(grayscale.flatten()))
    ranges = np.array_split(flatten, 4)
    palette = []
    for r in ranges:
        mean = np.mean(r)
        palette.append(mean)

    bounding_values = [0]
    for x in range(1, len(palette)):
        bounding_values.append(np.mean([palette[x - 1], palette[x]]))
    bounding_values.append(255)
    # encode image using 4 color palette
    for x in range(1, len(bounding_values)):
        grayscale[(grayscale >= bounding_values[x-1]) &
                  (grayscale <= bounding_values[x])] = palette[x-1]

    # threshold all values smaller than maximal
    grayscale[abs(grayscale - palette[-1]) < 2] = 255
    grayscale[grayscale < palette[-1]] = 0

    start_point = (grayscale.shape[0]/2, grayscale.shape[1]/2)
    Image.fromarray(grayscale.astype(np.uint8)).show()

    grayscale = erosion(grayscale)
    grayscale = erosion(grayscale)
    grayscale = erosion(grayscale)

    Image.fromarray(grayscale.astype(np.uint8)).show()

    grayscale = dylatation(grayscale)
    grayscale = dylatation(grayscale)

    Image.fromarray(grayscale.astype(np.uint8)).show()

    idx_l = start_point[0]
    gray_points = []

    while idx_l > 0:
        if(grayscale[idx_l, start_point[1]] == 255):
            gray_points.append((idx_l, start_point[1]))
            if len(gray_points) > 4:
                break
        else:
            gray_points = []

        idx_l -= 1

    idx_r = start_point[1]
    gray_points = []

    while idx_r < grayscale.shape[0]:
        if(grayscale[idx_r, start_point[1]] == 255):
            gray_points.append((idx_r, start_point[1]))
            if len(gray_points) > 4:
                break
        else:
            gray_points = []

        idx_r -= 1


if __name__ == "__main__":
    main({})
