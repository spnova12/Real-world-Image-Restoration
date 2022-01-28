import os
import argparse
import sys

import yaml

import rawRGB.preprocessing_ as rawRGBp
import sRGB.preprocessing_ as sRGBp


parser = argparse.ArgumentParser(description='Init')
parser.add_argument('-mode', default='None', type=str)
parser.add_argument('-train_align_net', action='store_true')
parser.add_argument('-exp_name', default='rain001', type=str)
parser.add_argument('-noise_type', default='R', type=str)
parser.add_argument('-cuda_num', default=None, type=str)
args = parser.parse_args()

# read yaml.
with open('dirs.yaml') as f:
    dirs = yaml.load(f, Loader=yaml.FullLoader)


# sRGB.
# Check DB
if args.mode == 'sRGB' and not args.train_align_net:
    sRGBp.preprocessing(dirs['sRGB']['DB_dir'])
# Train align network.
elif args.mode == 'sRGB' and args.train_align_net:
    # noise type : ('R', 'F', 'D', 'S', 'L' : Rain, Fog, Dust, Snow, Lowlight)
    # i f cuda_num is None it means use multi gpus.
    sRGBp.train_align_net(args.exp_name, dirs['sRGB']['DB_dir'], args.noise_type, args.cuda_num)


# rawRGB.
elif args.mode == 'rawRGB':
    rawRGBp.preprocessing(dirs['rawRGB']['DB_dir'])


# Error.
else:
    sys.exit('mode is not correct')




