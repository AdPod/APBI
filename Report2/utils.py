import numpy as np


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


def dilate(img_arr, kernel_size=5):
    r = int((kernel_size - 1)/2)
    working = bw_padding(img_arr, r)
    for x in range(img_arr.shape[0]):
        for y in range(img_arr.shape[1]):
            if (np.sum(working[x:x+kernel_size, y:y+kernel_size]) - working[x+r, y+r]) > 0:
                img_arr[x, y] = 255
    return img_arr


def erode(img_arr, kernel_size=5):
    r = int((kernel_size - 1)/2)
    s = 255*(kernel_size**2 - 1)
    working = bw_padding(img_arr, r)
    for x in range(img_arr.shape[0]):
        for y in range(img_arr.shape[1]):
            if (np.sum(working[x:x+kernel_size, y:y+kernel_size]) - working[x+r, y+r]) < s:
                img_arr[x, y] = 0
    return img_arr


def quantize(grayscale_img_arr, palette_size):
    img = np.copy(grayscale_img_arr)
    flatten = np.array(sorted(img.flatten()))
    ranges = np.array_split(flatten, palette_size)
    palette = []

    for r in ranges:
        mean = np.mean(r)
        palette.append(mean)

    bounding_values = [0]
    for x in range(1, len(palette)):
        bounding_values.append(np.mean([palette[x - 1], palette[x]]))
    bounding_values.append(255)

    # encode image using created color palette
    for x in range(1, len(bounding_values)):
        img[(img >= bounding_values[x-1]) &
            (img <= bounding_values[x])] = palette[x-1]
    return img, palette
