import os
import argparse
import sys

import yaml

import rawRGB.get_data_info_ as rawRGB_get_data_info
import sRGB.get_data_info_ as sRGB_get_data_info
import sRGB.folder_to_video as folder_to_video


parser = argparse.ArgumentParser(description='Get data info')
parser.add_argument('-mode', default='None', type=str)
parser.add_argument('-img_name_to_find', default=None, type=str)

parser.add_argument('-input_folder_dir', default=None, type=str)
parser.add_argument('-out_folder_dir', default=None, type=str)
parser.add_argument('-video_name', default=None, type=str)


args = parser.parse_args()

# read yaml.
with open('dirs.yaml') as f:
    dirs = yaml.load(f, Loader=yaml.FullLoader)


if args.mode == 'sRGB':
    sRGB_get_data_info.get_data_info(dirs['sRGB']['DB_dir'])
elif args.mode == 'sRGB_sampling':
    sRGB_get_data_info.get_data_info_and_samples(dirs['sRGB']['DB_dir'], args.img_name_to_find)
elif args.mode == 'sRGB_sample_to_video':
    folder_to_video.folder_to_video_sliding(args.input_folder_dir, args.out_folder_dir, args.video_name)



elif args.mode == 'rawRGB':
    rawRGB_get_data_info.get_data_info(dirs['rawRGB']['DB_dir'])

else:
    sys.exit('mode is not correct')



