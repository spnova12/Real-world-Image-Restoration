import os
import argparse
import sys

import yaml

import rawRGB.preprocessing_ as rawRGBp
import sRGB.preprocessing_ as sRGBp


parser = argparse.ArgumentParser(description='Init')
parser.add_argument('-mode', default='None', type=str)
args = parser.parse_args()

# read yaml.
with open('dirs.yaml') as f:
    dirs = yaml.load(f, Loader=yaml.FullLoader)


if args.mode == 'sRGB':
    sRGBp.preprocessing(dirs['sRGB']['DB_dir'])


elif args.mode == 'rawRGB':
    rawRGBp.preprocessing(dirs['rawRGB']['DB_dir'])

else:
    sys.exit('mode is not correct')




