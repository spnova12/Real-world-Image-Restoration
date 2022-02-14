import os
import argparse
import sys

import yaml

import rawRGB.get_data_info_ as rawRGB_get_data_info
import sRGB.get_data_info_ as sRGB_get_data_info


parser = argparse.ArgumentParser(description='Get data info')
parser.add_argument('-mode', default='None', type=str)
args = parser.parse_args()

# read yaml.
with open('dirs.yaml') as f:
    dirs = yaml.load(f, Loader=yaml.FullLoader)


if args.mode == 'sRGB':
    sRGB_get_data_info.get_data_info(dirs['sRGB']['DB_dir'])
elif args.mode == 'sRGB_sampling':
    sRGB_get_data_info.get_data_info2(dirs['sRGB']['DB_dir'])

elif args.mode == 'rawRGB':
    rawRGB_get_data_info.get_data_info(dirs['rawRGB']['DB_dir'])

else:
    sys.exit('mode is not correct')



