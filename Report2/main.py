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


def follow_region_border(regions, x, y, filled):
    border = set()
    _, img_h = regions.shape
    region_name = regions[x, y]
    line = y
    r_prev = None
    l_prev = None

    while line < img_h:
        row = list(regions[:, line])
        try:
            l = row.index(region_name)
            r = len(row) - row[::-1].index(region_name)
            border.add((l, line))
            border.add((r, line))

            filled[l, line] = 20
            filled[r, line] = 20
            filled[l+1:r, line] = region_name

            if l_prev != None and l_prev != l:
                for a in range(min(l_prev, l), max(l_prev, l)):
                    border.add((a, line))
                    filled[a, line] = 20

            if r_prev != None and r_prev != r:
                for a in range(min(r_prev, r), max(r_prev, r)):
                    border.add((a, line))
                    filled[a, line] = 20

            l_prev = l
            r_prev = r

        except ValueError:
            if l_prev != None and r_prev != None and l_prev != r_prev:
                for a in range(l_prev, r_prev):
                    border.add((a, line + 1))
                    filled[a, line + 1] = 20
            break
        line -= 1

    # shape starts with straight line

    r_prev = None
    l_prev = None
    line = y + 1
    while line < img_h:
        row = list(regions[:, line])
        try:
            l = row.index(region_name)
            r = len(row) - row[::-1].index(region_name)
            border.add((l, line))
            border.add((r, line))

            filled[l, line] = 20
            filled[r, line] = 20
            filled[l+1:r, line] = region_name

            if l_prev != None and l_prev != l:
                for a in range(min(l_prev, l), max(l_prev, l)):
                    border.add((a, line))
                    filled[a, line] = 20

            if r_prev != None and r_prev != r:
                for a in range(min(r_prev, r), max(r_prev, r)):
                    border.add((a, line))
                    filled[a, line] = 20

            l_prev = l
            r_prev = r

        except ValueError:
            if l_prev != None and r_prev != None and l_prev != r_prev:
                for a in range(l_prev, r_prev):
                    border.add((a, line - 1))
                    filled[a, line - 1] = 20
            break
        line += 1

    return border


def outline_regions(regions):
    filled = np.array(regions)
    outlined_regions = {}
    for x in range(regions.shape[0]):
        for y in range(regions.shape[1]):
            if regions[x, y] != 0 and not regions[x, y] in outlined_regions:
                name = regions[x, y]
                l = follow_region_border(regions, x, y, filled)
                sum_x = sum_y = x_max = y_max = 0
                x_min = regions.shape[0]
                y_min = regions.shape[1]
                for c_x, c_y in l:
                    sum_x += c_x
                    sum_y += c_y
                    if x_max < c_x:
                        x_max = c_x
                    if y_max < c_y:
                        y_max = c_y
                    if x_min > c_x:
                        x_min = c_x
                    if y_min > c_y:
                        y_min = c_y

                outlined_regions[name] = {
                    'list': l,
                    'name': name,
                    'count': len(l),
                    'center': (sum_x/len(l), sum_y/len(l)),
                    'x_max': x_max,
                    'y_max': y_max,
                    'x_min': x_min,
                    'y_min': y_min
                }
    return outlined_regions, filled


def find_regions(img_arr):
    img = bw_padding(img_arr, 1)
    regions = np.zeros_like(img)
    regions_count = 0

    regions_conflicts = set()

    for x in range(1, img.shape[0] - 1):
        for y in range(1, img.shape[1] - 1):
            if img[x, y] == 0:
                l = regions[x - 1, y]
                r = regions[x, y - 1]
                if l > 1 and r > 1:
                        # conflict
                    if l == r:
                        regions[x, y] = l
                    else:
                        sm = min(l, r)
                        lg = max(l, r)
                        regions_conflicts.add((sm, lg))
                        regions[x, y] = sm
                elif max(l, r) > 1:
                    regions[x, y] = max(l, r)
                else:
                    regions_count += 1
                    regions[x, y] = regions_count

    # join conflicts
    regions_conflicts = list(regions_conflicts)
    regions_conflicts = sorted(
        regions_conflicts, key=lambda a: a[0], reverse=True)

    removed = set()
    for conflict in regions_conflicts:
        if not conflict[1] in removed:
            regions[regions == conflict[1]] = conflict[0]
            regions_count -= 1
            removed.add(conflict[1])

    pure = np.array(regions)
    scaled = regions * 255 / regions_count
    scaled[scaled > 255] = 255
    scaled[scaled < 0] = 0
    return pure, scaled.round()


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
    # img = im_to_arr(args.img)
    img = im_to_arr('/home/dominik/Stud/Biometria/Report2/images/eye.jpg')

    img = filter_img(img, 'gauss')
    Image.fromarray(img.astype(np.uint8)).show()
    img = filter_img(img, 'gauss')
    Image.fromarray(img.astype(np.uint8)).show()

    img = expand_hist(img)
    Image.fromarray(img.astype(np.uint8)).show()
    img = contrast(img, 1.1)
    Image.fromarray(img.astype(np.uint8)).show()
    img = threshold(img, 10)
    Image.fromarray(img.astype(np.uint8)).show()
    # dylated = np.array(img)
    # Image.fromarray(dylated.astype(np.uint8)).show()

    # eroded = np.array(img)
    dylated = dylatation(img)
    Image.fromarray(dylated.astype(np.uint8)).show()
    dylated = dylatation(dylated)
    Image.fromarray(dylated.astype(np.uint8)).show()

    eroded = erosion(dylated)
    Image.fromarray(eroded.astype(np.uint8)).show()
    eroded = erosion(eroded)
    Image.fromarray(eroded.astype(np.uint8)).show()

    regions, printable_regions = find_regions(eroded)
    Image.fromarray(printable_regions.astype(np.uint8)).show()
    # regions, printable_regions = find_regions(img)
    regions_areas = np.unique(regions, return_counts=True)

    scaled = regions * 255 / len(regions_areas[0])
    scaled[scaled > 255] = 255
    scaled[scaled < 0] = 0

    scaled = Image.fromarray(scaled.astype(np.uint8))
    scaled.show()

    outlined_regions, filled = outline_regions(regions)
    tmp_sort_list = []

    for o in outlined_regions:
        tmp_sort_list.append(
            (outlined_regions[o]['name'], outlined_regions[o]['count']))

    tmp_sort_list = sorted(tmp_sort_list, key=lambda a: a[1], reverse=True)

    total_img_area = img.shape[0]*img.shape[1]

    guess_for_pupil = None

    for o in tmp_sort_list:
        ol = outlined_regions[o[0]]

        area = regions_areas[1][list(regions_areas[0]).index(ol['name'])]

        if area < 0.01*total_img_area:
            continue

        print("\n")
        r1 = math.sqrt(area / math.pi)
        print("r1: " + str(r1))

        r2 = (ol['y_max'] - ol['y_min'])/2
        r3 = ol['count'] / 2 / math.pi

        print("r2: " + str(r2))
        print("r3: " + str(r3))
        dist_from_img_center = math.sqrt(math.pow(
            img.shape[0]/2 - ol['center'][0], 2) + math.pow(img.shape[1]/2 - ol['center'][1], 2))
        print("distance from center:  " + str(dist_from_img_center))
        if abs(r1 - r2) < 0.2*r1 and abs(r1 - r3) < 0.2*r1 and dist_from_img_center < 0.2*img.shape[1]:
            guess_for_pupil = ol
            break

    scaled = filled * 255 / len(regions_areas[0])
    scaled[scaled > 255] = 255
    scaled[scaled < 0] = 0

    scaled = Image.fromarray(scaled.astype(np.uint8))
    scaled.show()

    if guess_for_pupil == None:
        return

    mask = Image.new('RGBA', (img.shape[1], img.shape[0]))
    draw = ImageDraw.Draw(mask)
    draw.ellipse((guess_for_pupil['y_min'], guess_for_pupil['x_min'], guess_for_pupil['y_max'], guess_for_pupil['x_max']), fill='blue', outline='blue')

    mask.show()
    # printable_regions = Image.fromarray(printable_regions.astype(np.uint8))
    # printable_regions.show()

    # for x in range(img.shape[0]):
    #     for y in range(img.shape[1]):
    #         r, g, b = img[x, y]
    #         if (r < 1 and g < 1 and b < 1):
    #             img[x, y] = 249, 20, 145

    # img = Image.fromarray(img.astype(np.uint8))
    # dylated = Image.fromarray(dylated.astype(np.uint8))
    # eroded = Image.fromarray(eroded.astype(np.uint8))
    # eroded.save('images/eroded.png')
    # img.show()
    # dylated.show()
    # eroded.show()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="iris segmentation")
    # parser.add_argument('-i', '--img', required=True)
    # args = parser.parse_args()
    # main(args)
    main({})


# https://web.cs.wpi.edu/~emmanuel/courses/cs545/S14/slides/lecture08.pdf
