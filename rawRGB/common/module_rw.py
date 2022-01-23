import cv2
import gzip
import pickle
import bz2
import lzma
import os
import sys


def write_obj(filename, obj, mode):
    # mode : png, pkl, gz, bz2, lzma
    if mode == 'png':
        cv2.imwrite(f'{filename}.png', obj)
    elif mode == 'pkl':
        with open(f'{filename}', 'wb') as f:
            pickle.dump(obj, f)
    elif mode == 'gz':
        with gzip.open(f'{filename}.gz', "wb") as f:
            pickle.dump(obj, f)
    elif mode == 'bz2':
        with bz2.BZ2File(f'{filename}.bz2', 'wb') as f:
            pickle.dump(obj, f)
    elif mode == 'lzma':
        with lzma.open(f'{filename}.xz', 'wb') as f:
            pickle.dump(obj, f)


def read_obj(filename):
    if os.path.isfile(filename):
        mode = os.path.splitext(filename)[1]
        # mode : .png, .pkl, .gz, .bz2, .lzma
        if mode == '.png':
            return cv2.imread(f'{filename}')
        elif mode == '.pkl':
            with open(f'{filename}', 'rb') as f:
                return pickle.load(f)
        elif mode == '.gz':
            with gzip.open(f'{filename}', "rb") as f:
                return pickle.load(f)
        elif mode == '.bz2':
            with bz2.BZ2File(f'{filename}', 'rb') as f:
                return pickle.load(f)
        elif mode == '.lzma':
            with lzma.open(f'{filename}', 'rb') as f:
                return pickle.load(f)
    else:
        sys.exit('No file')
