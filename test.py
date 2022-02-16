import os
import argparse
import sys

import yaml

import rawRGB.test as rawRGB_test
import sRGB.test as sRGB_test


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-mode', default='None', type=str)
parser.add_argument('-my_db', action='store_true')
parser.add_argument('-cuda_num', default=None, type=str)
parser.add_argument('-noise_type', default='R', type=str)

parser.add_argument('-input_folder_dir', default='my_out', type=str)
parser.add_argument('-out_folder_name', default='my_out', type=str)

args = parser.parse_args()

# read yaml.
with open('dirs.yaml') as f:
    dirs = yaml.load(f, Loader=yaml.FullLoader)


# python test.py -mode sRGB -noise_type R -out_dir_name my_R -my_db -cuda_num 0

if args.mode == 'sRGB':

    # option 1. Get final test result (get psnr and save some samples)
    if not args.my_db:
        sRGB_test.test(
            dirs['pretrain_net_dir_for_test'][args.noise_type],
            dirs['pretrain_net_dir_for_align'][args.noise_type],
            dirs['sRGB']['DB_dir'],
            args.noise_type,
            args.cuda_num
        )

    # option 2. Inference some samples and save (No require GT)
    else:
        sRGB_test.test_my_db(
            dirs['pretrain_net_dir_for_test'][args.noise_type],
            args.input_folder_dir, args.out_folder_name, args.cuda_num)




#
#
# ##################################################################################################################
# elif args.mode == 'rawRGB':
#     rawRGB_train.train(args.exp_name, dirs['rawRGB']['DB_dir'], args.cuda_num)
#
# else:
#     sys.exit('mode is not correct')
