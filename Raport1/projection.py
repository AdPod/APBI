#!/usr/bin/python
from PIL import Image
import numpy as np
import sys
from utils import bound, im_to_arr
from grayscale import conv_gray
from threshold import threshold


def main(args):

    img = im_to_arr(args[0])

    thresholded = threshold(img, 150)

    projection = np.zeros(
        (thresholded.shape[0] + 32, thresholded.shape[1] + 32))
    projection[32:, 32:] = thresholded
    projection = np.repeat(projection, 3).reshape(
        (projection.shape[0], projection.shape[1], 3))

    projection[30:32, :] = (0, 0, 200)
    projection[:, 30:32] = (0, 0, 200)

    for x in range(thresholded.shape[0]):
        sum_in_row = np.sum(thresholded[x, :])
        white_ratio = sum_in_row/thresholded.shape[1]/255
        stretched = int(round(30*white_ratio))

        for p in range(stretched, -1, -1):
            projection[x + 32, p] = (255, 0, 0)

    for x in range(thresholded.shape[1]):
        sum_in_col = np.sum(thresholded[:, x])
        white_ratio = sum_in_col/thresholded.shape[0]/255
        stretched = int(round(30*white_ratio))
        for p in range(stretched, -1, -1):
            projection[p, x + 32] = (255, 0, 0)

    projection = Image.fromarray(projection.astype(np.uint8))
    projection.show()


if __name__ == "__main__":
    main(sys.argv[1:])
