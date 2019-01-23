from algorithms.ta import ThinningAlgorithm
import numpy as np

A0 = [3, 6, 7, 12, 14, 15, 24, 28, 30, 31, 48, 56, 60,
      62, 63, 96, 112, 120, 124, 126, 127, 129, 131, 135,
      143, 159, 191, 192, 193, 195, 199, 207, 223, 224,
      225, 227, 231, 239, 240, 241, 243, 247, 248, 249,
      251, 252, 253, 254]
A1 = [7, 14, 28, 56, 112, 131, 193, 224]
A2 = [7, 14, 15, 28, 30, 56, 60, 112, 120, 131, 135,
      193, 195, 224, 225, 240]
A3 = [7, 14, 15, 28, 30, 31, 56, 60, 62, 112, 120,
      124, 131, 135, 143, 193, 195, 199, 224, 225, 227,
      240, 241, 248]
A4 = [7, 14, 15, 28, 30, 31, 56, 60, 62, 63, 112, 120,
      124, 126, 131, 135, 143, 159, 193, 195, 199, 207,
      224, 225, 227, 231, 240, 241, 243, 248, 249, 252]
A5 = [7, 14, 15, 28, 30, 31, 56, 60, 62, 63, 112, 120,
      124, 126, 131, 135, 143, 159, 191, 193, 195, 199,
      207, 224, 225, 227, 231, 239, 240, 241, 243, 248,
      249, 251, 252, 254]
A1_pix = [3, 6, 7, 12, 14, 15, 24, 28, 30, 31, 48, 56,
          60, 62, 63, 96, 112, 120, 124, 126, 127, 129, 131,
          135, 143, 159, 191, 192, 193, 195, 199, 207, 223,
          224, 225, 227, 231, 239, 240, 241, 243, 247, 248,
          249, 251, 252, 253, 254]

A = [A1, A2, A3, A4, A5]

def remove_borders(img):
    removed_pixels = 0
    for a in A:
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                # check if pixel is border
                if img[x, y] != 2:
                    continue
                nei = (img[x-1:x+2, y-1:y+2] != 0).flatten()
                weighted_sum = np.sum(
                    np.array([128, 1, 2, 64, 0, 4, 32, 16, 8]) * nei)
                if weighted_sum in a:
                    img[x, y] = 0
                    removed_pixels += 1
    return img, removed_pixels


def one_pixel_thin(img):
    tmp = np.zeros_like(img)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
                # check if pixel is border
            if img[x, y] == 0:
                continue

            nei = (img[x-1:x+2, y-1:y+2] != 0).flatten()
            weighted_sum = np.sum(
                np.array([128, 1, 2, 64, 0, 4, 32, 16, 8]) * nei)
            if weighted_sum not in A1_pix:
                tmp[x, y] = 1
    return tmp


def mark_borders(img):
    tmp = np.zeros_like(img)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x, y] == 0:
                continue
            nei = (img[x-1:x+2, y-1:y+2] != 0).flatten()
            weighted_sum = np.sum(
                np.array([128, 1, 2, 64, 0, 4, 32, 16, 8]) * nei)
            tmp[x, y] = 2 if weighted_sum in A0 else 1
    return tmp


class K3M(ThinningAlgorithm):
    def __init__(self, img_arr, show_steps, save_steps, save_location='./'):
        ThinningAlgorithm.__init__(
            self, img_arr, show_steps, save_steps, save_location, 'K3M')

    def thin(self):    
        removed_pixels = 1
        iters = 0
        while removed_pixels > 0:
            iters += 1
            self.img_arr = mark_borders(self.img_arr)
            self.img_arr, removed_pixels = remove_borders(self.img_arr)
            self.debug(f'iteration_{iters}')



        self.img_arr = one_pixel_thin(self.img_arr)
        self.show_steps = True
        self.debug('result')

        print(f'K3M: thinned in: {iters} iters')
        return self.img_arr

