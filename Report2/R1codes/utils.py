from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def bound(arr):
    return np.clip(arr, 0, 255)


def im_to_arr(path):
    try:
        with Image.open(path) as image:
            im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
            if im_arr.size == image.size[0]*image.size[1]*3:
                im_arr = im_arr.reshape(
                    (image.size[1], image.size[0], 3)).astype(np.int16)
            elif im_arr.size == image.size[0]*image.size[1]:
                im_arr = im_arr.reshape(
                    (image.size[1], image.size[0])).astype(np.int16)
        return im_arr
    except FileNotFoundError:
        print('remember to pass correct filepath as argument')


def occur_tab(arr):
    occur_tab = np.zeros(256)
    unique, counts = np.unique(arr, return_counts=True)
    occur_tab[unique] = counts
    return occur_tab


def print_hist_img(imgs):

    if not isinstance(imgs, list):
        imgs = [imgs]

    for img in imgs:
        red_occur_tab = occur_tab(img[:, :, 0].flatten())
        green_occur_tab = occur_tab(img[:, :, 1].flatten())
        blue_occur_tab = occur_tab(img[:, :, 2].flatten())

        print_hist_occur_tab(red_occur_tab, green_occur_tab, blue_occur_tab)

    plt.show()


def print_hist_occur_tab(red_occur_tab, green_occur_tab, blue_occur_tab):
    fig = plt.figure(figsize=(10, 3))

    fig.add_subplot(1, 3, 1)
    plt.bar(np.arange(len(red_occur_tab)), red_occur_tab, color='red')
    plt.title('red color')

    fig.add_subplot(1, 3, 2)
    plt.bar(np.arange(len(green_occur_tab)), green_occur_tab, color='green')
    plt.title('green color')

    fig.add_subplot(1, 3, 3)
    plt.bar(np.arange(len(blue_occur_tab)), blue_occur_tab, color='blue')
    plt.title('blue color')
