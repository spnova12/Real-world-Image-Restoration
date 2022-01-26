import os
import argparse
import sys

import yaml

import rawRGB.get_data_info_ as getdatainfo
# import sRGB.preprocessing_ as sRGBp


parser = argparse.ArgumentParser(description='Init')
parser.add_argument('-mode', default='None', type=str)
args = parser.parse_args()

# read yaml.
with open('dirs.yaml') as f:
    dirs = yaml.load(f, Loader=yaml.FullLoader)


if args.mode == 'sRGB':
    pass


elif args.mode == 'rawRGB':
    getdatainfo.get_data_info(dirs['rawRGB']['DB_dir'])

else:
    sys.exit('mode is not correct')



