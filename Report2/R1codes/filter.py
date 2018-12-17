#!/usr/bin/python
import argparse
import sys

import numpy as np
from PIL import Image

from .grayscale import conv_gray
from .utils import bound, im_to_arr

averaging = np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]])
gauss = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])
sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

sobel0 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel45 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
sobel90 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
sobel135 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])

roberts1 = np.array([[1, 0], [0, -1]])
roberts2 = np.array([[0, 1], [-1, 0]])


def padding(arr):
    padded = np.zeros((arr.shape[0] + 2, arr.shape[1] + 2, 3))
    padded[1:-1, 1:-1, :] = arr
    # mirror padding
    padded[0, :, :] = padded[1, :, :]
    padded[-1, :, :] = padded[-2, :, :]
    padded[:, 0, :] = padded[:, 0, :]
    padded[:, -1, :] = padded[:, -2, :]
    return padded


def applyFilter(arr, filter):
    padded = padding(arr)
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            padded[x+1, y+1, 0] = bound(np.sum(np.multiply(
                padded[x:x+3, y:y+3, 0], filter)))
            padded[x+1, y+1, 1] = bound(np.sum(np.multiply(
                padded[x:x+3, y:y+3, 1], filter)))
            padded[x+1, y+1, 2] = bound(np.sum(np.multiply(
                padded[x:x+3, y:y+3, 2], filter)))

    return padded


def sobel_filter(arr):
    padded = padding(arr)

    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            r, g, b = 0, 0, 0

            r += np.abs(np.sum(np.multiply(padded[x:x+3, y:y+3, 0], sobel0)))
            g += np.abs(np.sum(np.multiply(padded[x:x+3, y:y+3, 1], sobel0)))
            b += np.abs(np.sum(np.multiply(padded[x:x+3, y:y+3, 2], sobel0)))

            r += np.abs(np.sum(np.multiply(padded[x:x+3, y:y+3, 0], sobel45)))
            g += np.abs(np.sum(np.multiply(padded[x:x+3, y:y+3, 1], sobel45)))
            b += np.abs(np.sum(np.multiply(padded[x:x+3, y:y+3, 2], sobel45)))

            r += np.abs(np.sum(np.multiply(padded[x:x+3, y:y+3, 0], sobel90)))
            g += np.abs(np.sum(np.multiply(padded[x:x+3, y:y+3, 1], sobel90)))
            b += np.abs(np.sum(np.multiply(padded[x:x+3, y:y+3, 2], sobel90)))

            r += np.abs(np.sum(np.multiply(padded[x:x+3, y:y+3, 0], sobel135)))
            g += np.abs(np.sum(np.multiply(padded[x:x+3, y:y+3, 1], sobel135)))
            b += np.abs(np.sum(np.multiply(padded[x:x+3, y:y+3, 2], sobel135)))

            padded[x, y] = bound([int(r), int(g), int(b)])

    return padded


def roberts_filter(arr):
    padded = padding(arr)

    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            padded[x, y, 0] = bound(np.absolute(np.sum(np.multiply(
                padded[x:x+2, y:y+2, 0], roberts1))) + np.absolute(np.sum(np.multiply(padded[x:x+2, y:y+2, 0], roberts2))))
            padded[x, y, 1] = bound(np.absolute(np.sum(np.multiply(
                padded[x:x+2, y:y+2, 1], roberts1))) + np.absolute(np.sum(np.multiply(padded[x:x+2, y:y+2, 1], roberts2))))
            padded[x, y, 2] = bound(np.absolute(np.sum(np.multiply(
                padded[x:x+2, y:y+2, 2], roberts1))) + np.absolute(np.sum(np.multiply(padded[x:x+2, y:y+2, 2], roberts2))))

    return padded


def fltr(img_arr, filterType):
    if filterType == 'averaging':
        return applyFilter(img_arr, averaging)
    if filterType == 'gauss':
        return applyFilter(img_arr, gauss)
    if filterType == 'sharpening':
        return applyFilter(img_arr, sharpening)
    if filterType == 'roberts':
        return roberts_filter(img_arr)
    if filterType == 'sobel':
        return sobel_filter(img_arr)


def main(args):

    img = im_to_arr(args.img)

    transtormed = fltr(img, args.filter)
    transtormed = Image.fromarray(transtormed.astype(np.uint8))
    transtormed.show()

    img = Image.fromarray(img.astype(np.uint8))
    img.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="apply filters")
    parser.add_argument(
        '-f', '--filter', choices=['averaging', 'gauss', 'sharpening', 'roberts', 'sobel'], default='averaging')
    parser.add_argument('-i', '--img', required=True)
    args = parser.parse_args()
    main(args)
