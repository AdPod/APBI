import argparse
from tools.utils import im_to_arr, bw_padding
from tools.grayscale import conv_gray
import numpy as np
from PIL import Image

# crop image to reduce redundant calcuations

to_del = [3, 5, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 48, 52, 53, 54, 55, 56, 60, 61, 62, 63, 65, 67, 69, 71, 77, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89, 91, 92, 93, 94, 95, 97, 99, 101, 103, 109, 111, 112, 113, 115, 116, 117, 118, 119, 120, 121, 123, 124, 125, 126,
          127, 131, 133, 135, 141, 143, 149, 151, 157, 159, 181, 183, 189, 191, 192, 193, 195, 197, 199, 205, 207, 208, 209, 211, 212, 213, 214, 215, 216, 217, 219, 220, 221, 222, 223, 224, 225, 227, 229, 231, 237, 239, 240, 241, 243, 244, 245, 246, 247, 248, 249, 251, 252, 253, 254, 255]

# array of all combinations of weights 2,3,4 sticking neighbours may produce
mark_as_4 = [3, 6, 7, 12, 14, 15, 24, 28, 30, 48, 56, 60, 96,
             112, 120, 129, 131, 135, 192, 193, 195, 224, 225, 240]


def crop(img):
    non_zero = np.transpose(img.nonzero())
    ma = np.max(non_zero, axis=0)  # max row and column of nonzero
    mi = np.min(non_zero, axis=0)
    cropped = img[mi[0]:ma[0], mi[1]:ma[1]]
    cropped = bw_padding(cropped, 1, 0)
    return cropped


def thin(img):
    shape_size = 0
    # mark pixels touching contour as 2, 3 or 4
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x, y] == 0:
                continue
            nei = (img[x-1:x+2, y-1:y+2] == 0).flatten()
            weighted_sum = np.sum(
                np.array([128, 1, 2, 64, 0, 4, 32, 16, 8]) * ~nei)
            if weighted_sum in mark_as_4:
                img[x, y] = 4
            elif any(o for o in nei[1::2]):
                img[x, y] = 2
            elif any(e for e in nei[::2]):
                img[x, y] = 3

    # delete all pixels marked as 4
    img[img == 4] = 0
    # working = np.array(img, copy=True)

    for N in [2, 3]:
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if not img[x, y] == N:
                    continue
                nei = (img[x-1:x+2, y-1:y+2] != 0).flatten()
                weighted_sum = np.sum(
                    np.array([128, 1, 2, 64, 0, 4, 32, 16, 8]) * nei)

                if weighted_sum in to_del:
                    img[x, y] = 0
                else:
                    img[x, y] = 1
                    shape_size += 1
    return img, shape_size


def is_line(img):
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x, y] == 0:
                continue
            nei = img[x-1:x+2, y-1:y+2]
            nei = np.concatenate((nei[0, :], nei[1, 2],
                                  nei[2, :], nei[1, 0], nei[0, 0]), axis=None)
            for n in range(8):
                if nei[n] > 0 and nei[n + 1] > 0:
                    return False
    return True


def main(args):
    # img = im_to_arr(args.img)
    img = im_to_arr(
        'C:/Users/tukan/Documents/SharedProjects/Biometria/APBI/Report3/images/Yaa.PNG')
    img, _ = conv_gray(img)

    img[img < 128] = 1
    img[img >= 127] = 0
    img = crop(img)

    Image.fromarray(img.astype(np.int8)*60).show()

    shape_size = np.count_nonzero(img)
    iters = 0
    while not is_line(img):
        iters += 1
        img, new_shape_size = thin(img)
        # check if shape's size was reduced
        Image.fromarray(img.astype(np.int8)*60).show()
        if shape_size == new_shape_size:
            break
        shape_size = new_shape_size

    print(f'thinned in: {iters} iters')

    # thr = np.zeros_like(img)
    # thr[img == 0] = 255
    # Image.fromarray(thr.astype(np.int8)).show()
    # thr = np.zeros_like(img)
    # thr[img == 1] = 255
    # Image.fromarray(thr.astype(np.int8)).show()
    # thr = np.zeros_like(img)
    # thr[img == 2] = 255
    # Image.fromarray(thr.astype(np.int8)).show()
    # thr = np.zeros_like(img)
    # thr[img == 3] = 255
    # Image.fromarray(thr.astype(np.int8)).show()
    # thr = np.zeros_like(img)
    # thr[img == 4] = 255
    # Image.fromarray(thr.astype(np.int8)).show()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="skeletonization")
    # parser.add_argument('-i', '--img', required=True)
    # args = parser.parse_args()
    # main(args)
    main([])
