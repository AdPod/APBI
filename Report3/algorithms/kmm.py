from algorithms.ta import ThinningAlgorithm
import numpy as np

to_del = [3, 5, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 48, 52, 53, 54, 55, 56, 60, 61, 62, 63, 65, 67, 69, 71, 77, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89, 91, 92, 93, 94, 95, 97, 99, 101, 103, 109, 111, 112, 113, 115, 116, 117, 118, 119, 120, 121, 123, 124, 125, 126,
          127, 131, 133, 135, 141, 143, 149, 151, 157, 159, 181, 183, 189, 191, 192, 193, 195, 197, 199, 205, 207, 208, 209, 211, 212, 213, 214, 215, 216, 217, 219, 220, 221, 222, 223, 224, 225, 227, 229, 231, 237, 239, 240, 241, 243, 244, 245, 246, 247, 248, 249, 251, 252, 253, 254, 255]

# array of all combinations of weights 2,3,4 sticking neighbours may produce
mark_as_4 = [3, 6, 7, 12, 14, 15, 24, 28, 30, 48, 56, 60, 96,
             112, 120, 129, 131, 135, 192, 193, 195, 224, 225, 240]


def remove_border(img):
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
    print('it is line lol')
    return True


class KMM(ThinningAlgorithm):
    def __init__(self, img_arr, show_steps, save_steps, save_location='./'):
        ThinningAlgorithm.__init__(
            self, img_arr, show_steps, save_steps, save_location, 'KMM')

    def thin(self):
        shape_size = np.count_nonzero(self.img_arr)
        iters = 0

        not_hanged_count = 1
        while not is_line(self.img_arr):
            iters += 1
            self.img_arr, new_shape_size = remove_border(self.img_arr)

            if shape_size == new_shape_size:
                # hack, it sometimes changes without changing size
                not_hanged_count -= 1
                if not_hanged_count < 0:
                    break
            else:
                not_hanged_count = 1

            self.debug(f'iteration_{iters}')
            shape_size = new_shape_size

        self.show_steps = True
        self.debug('result')
        print(f'KMM: thinned in: {iters} iters')

        return self.img_arr
