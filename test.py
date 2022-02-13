import os
import argparse
import sys

import yaml

import rawRGB.test as rawRGB_test
import sRGB.test as sRGB_test


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-my_db', action='store_true')
parser.add_argument('-mode', default='None', type=str)
parser.add_argument('-cuda_num', default=None, type=str)
parser.add_argument('-noise_type', default='R', type=str)
parser.add_argument('-out_dir_name', default='my_out', type=str)

args = parser.parse_args()

# read yaml.
with open('dirs.yaml') as f:
    dirs = yaml.load(f, Loader=yaml.FullLoader)


# python test.py -mode sRGB -noise_type R -out_dir_name my_R -my_db -cuda_num 0

if args.mode == 'sRGB':
    # option 1. Get final test result (get psnr and save some samples)
    if not args.my_db:
        pass

    # option 2. Inference some samples and save (No require GT)
    else:
        cuda_num = args.cuda_num

        # Get the checkpoint weight.
        pretrain_net_dir_for_test = dirs['pretrain_net_dir_for_test'][f'{args.noise_type}']

        # Get the sample dirs for inference.
        # from folders.
        test_DB_folder_dir = dirs['test_DB_folder_dir'][f'{args.noise_type}']
        # from individually.
        test_DB_dirs = dirs['test_DB_dirs'][f'{args.noise_type}']
        test_DB_dirs = [x for x in test_DB_dirs if x is not None]

        test_DB_dir_list = None
        if test_DB_folder_dir:
            test_DB_dir_list = [os.path.join(test_DB_folder_dir, x) for x in sorted(os.listdir(test_DB_folder_dir))]
        else:
            print('Check yaml. The dir is empty.')
            quit()

        # final test sample list.
        test_DB_dir_list += test_DB_dirs

        sRGB_test.test_my_db(pretrain_net_dir_for_test, test_DB_dir_list, args.out_dir_name, cuda_num)




#
#
# ##################################################################################################################
# elif args.mode == 'rawRGB':
#     rawRGB_train.train(args.exp_name, dirs['rawRGB']['DB_dir'], args.cuda_num)
#
# else:
#     sys.exit('mode is not correct')
