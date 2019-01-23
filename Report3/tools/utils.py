from PIL import Image
import numpy as np


def bw_padding(arr, size, fill_color=None):
    padded = np.zeros((arr.shape[0] + 2*size, arr.shape[1] + 2*size))
    padded[size:-size, size:-size] = arr
    if fill_color != None:
        for x in range(size):
            padded[x, :] = fill_color
            padded[-1-x, :] = fill_color
            padded[:, x] = fill_color
            padded[:, -1-x] = fill_color
        return padded
    # mirror padding
    for x in range(size):
        padded[x, :] = padded[size, :]
        padded[-1-x, :] = padded[-1-size, :]
    for y in range(size):
        padded[:, y] = padded[:, size]
        padded[:, -1-y] = padded[:, -1-size]

    return padded

def im_to_arr(path):
    try:
        with Image.open(path) as image:
            im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
            im_arr = im_arr.reshape(
                (image.size[1], image.size[0], round(im_arr.size/image.size[0]/image.size[1]))).astype(np.int16)
            im_arr = im_arr[:, :, :3]
        return im_arr
    except FileNotFoundError:
        print('remember to pass correct filepath as argument. Image must be in rgb format')
