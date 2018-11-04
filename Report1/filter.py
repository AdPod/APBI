#!/usr/bin/python
import sys

import numpy as np
from PIL import Image

from grayscale import conv_gray
from utils import bound, im_to_arr

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
            padded[x, y, 0] = np.sum(np.multiply(
                padded[x:x+3, y:y+3, 0], filter))
            padded[x, y, 1] = np.sum(np.multiply(
                padded[x:x+3, y:y+3, 1], filter))
            padded[x, y, 2] = np.sum(np.multiply(
                padded[x:x+3, y:y+3, 2], filter))

    return padded


def sobel_filter(arr):
    padded = padding(arr)

    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            a = padded[x:x+3, y:y+3, :]
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
            padded[x, y, 0] = np.absolute(np.sum(np.multiply(
                padded[x:x+2, y:y+2, 0], roberts1))) + np.absolute(np.sum(np.multiply(padded[x:x+2, y:y+2, 0], roberts2)))
            padded[x, y, 1] = np.absolute(np.sum(np.multiply(
                padded[x:x+2, y:y+2, 1], roberts1))) + np.absolute(np.sum(np.multiply(padded[x:x+2, y:y+2, 1], roberts2)))
            padded[x, y, 2] = np.absolute(np.sum(np.multiply(
                padded[x:x+2, y:y+2, 2], roberts1))) + np.absolute(np.sum(np.multiply(padded[x:x+2, y:y+2, 2], roberts2)))

    return padded


def main(args):

    img = im_to_arr(args[0])

    avg = applyFilter(img, averaging)
    avg = Image.fromarray(avg.astype(np.uint8))
    avg.show()

    ga = applyFilter(img, gauss)
    ga = Image.fromarray(ga.astype(np.uint8))
    ga.show()
    
    sha = applyFilter(img, sharpening)
    sha = Image.fromarray(sha.astype(np.uint8))
    sha.show()


    rob = roberts_filter(img)
    rob = Image.fromarray(rob.astype(np.uint8))
    rob.show()

    sobel = sobel_filter(img)
    sobel = Image.fromarray(sobel.astype(np.uint8))
    sobel.show()

    img = Image.fromarray(img.astype(np.uint8))
    img.show()


if __name__ == "__main__":
    main(sys.argv[1:])
