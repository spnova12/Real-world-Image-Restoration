import os
import argparse
import sys

import yaml

import rawRGB.train as rawRGB_train
import sRGB.train as sRGB_train


parser = argparse.ArgumentParser(description='Init')
parser.add_argument('-mode', default='None', type=str)
parser.add_argument('-exp_name', default='rawRGB000', type=str)
args = parser.parse_args()

# read yaml.
with open('dirs.yaml') as f:
    dirs = yaml.load(f, Loader=yaml.FullLoader)


if args.mode == 'sRGB':
    sRGB_train.
    pass


elif args.mode == 'rawRGB':
    rawRGB_train.train(args.exp_name, dirs['rawRGB']['DB_dir'])

else:
    sys.exit('mode is not correct')
