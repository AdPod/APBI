import argparse
from tools.utils import im_to_arr, bw_padding
from tools.grayscale import conv_gray
import numpy as np
import re
from algorithms import k3m, kmm

# crop image to reduce redundant calcuations


def crop(img):
    non_zero = np.transpose(img.nonzero())
    ma = np.max(non_zero, axis=0)  # max row and column of nonzero
    mi = np.min(non_zero, axis=0)
    cropped = img[mi[0]:ma[0], mi[1]:ma[1]]
    cropped = bw_padding(cropped, 1, 0)
    return cropped


def main(args):
    img = im_to_arr(args.img)
    img, _ = conv_gray(img)

    img[img < 128] = 1
    img[img >= 127] = 0
    img = crop(img)

    img_kmm = np.copy(img)
    img_k3m = np.copy(img)
    del img

    if args.algorithm == 'both':
        kmm.KMM(img_kmm, args.show_steps, args.save_steps,
                args.save_location).thin()
        k3m.K3M(img_k3m, args.show_steps, args.save_steps,
                args.save_location).thin()
    elif args.algorithm == 'kmm':
        kmm.KMM(img_kmm, args.show_steps, args.save_steps,
                args.save_location).thin()
    elif args.algorithm == 'k3m':
        k3m.K3M(img_k3m, args.show_steps, args.save_steps,
                args.save_location).thin()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="skeletonization using KMM and K3M")
    parser.add_argument('-i', '--img', required=True)
    parser.add_argument('--show-steps', required=False,
                        default=False, type=str2bool)
    parser.add_argument('--save-steps', required=False,
                        default=False, type=str2bool)
    parser.add_argument('--save-location', required=False, default='./results/')
    parser.add_argument('-a', '--algorithm', required=False,
                        default='both', choices=['kmm', 'k3m', 'both'])
    args = parser.parse_args()
    main(args)
