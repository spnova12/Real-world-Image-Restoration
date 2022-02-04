import os
import argparse
import sys

import yaml

import rawRGB.train as rawRGB_train
import sRGB.train as sRGB_train


parser = argparse.ArgumentParser(description='Train')
parser.add_argument('-mode', default='None', type=str)
parser.add_argument('-exp_name', default='rawRGB000', type=str)
parser.add_argument('-cuda_num', default=None, type=str)

parser.add_argument('-noise_type', default='R', type=str)

args = parser.parse_args()

# read yaml.
with open('dirs.yaml') as f:
    dirs = yaml.load(f, Loader=yaml.FullLoader)


if args.mode == 'sRGB':
    sRGB_train.train(args.exp_name, dirs['sRGB']['DB_dir'], args.noise_type,
                     dirs['pretrain_net_dir_for_align'][args.noise_type], args.cuda_num)


elif args.mode == 'rawRGB':
    rawRGB_train.train(args.exp_name, dirs['rawRGB']['DB_dir'], args.cuda_num)

else:
    sys.exit('mode is not correct')
