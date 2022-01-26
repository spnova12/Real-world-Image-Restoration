import os
import glob
from rawRGB.common.module_init_raw_DB_utils import *
import argparse

def get_data_info(DB_dir):
    # Get this script's dir.
    preprocessing_dir = os.path.dirname(os.path.realpath(__file__))

    patch_DB_dir = DB_dir + '_patches'
    json_folder_dir = DB_dir


    # Get the paired list from DB_dir.
    raw_dir_paired_list = DB_dir_to_paired_list(patch_DB_dir, json_folder_dir)


    # Show the list info.
    print_wrap('Show the list info.')
    print('len(raw_dir_paired_list) :', len(raw_dir_paired_list))
    print('raw_dir_paired_list[0] :')
    for item in raw_dir_paired_list[0].items():
        print(f"{item[0]}: {item[1]}")


    # visualize .bz2 sample(16bit raw) to sRGB png.
    sample1 = raw_dir_paired_list[0]


    # input

    input_sample_dir = f'{preprocessing_dir}/input_patch_sample_sRGB_uint8.png'
    print_wrap(f'Write input sample. ({input_sample_dir})')
    input_cfa_data_patch = read_obj(sample1['input'])
    input_metadata_dict = read_obj(sample1['input_metadata_dict'])
    input_cfa_mask = read_obj(f"{os.path.splitext(sample1['input'])[0]}_cfa_mask.bz2")
    input_srgb_uint8 = raw_16bit_postprocess(input_cfa_data_patch, input_metadata_dict, input_cfa_mask)
    cv2.imwrite(input_sample_dir, input_srgb_uint8)


    # target
    target_sample_dir = f'{preprocessing_dir}/target_patch_sample_sRGB_uint8.png'
    print_wrap(f'Write target sample. ({target_sample_dir})')
    target__cfa_data_patch = read_obj(sample1['target'])
    target_metadata_dict = read_obj(sample1['target_metadata_dict'])
    target_cfa_mask = read_obj(f"{os.path.splitext(sample1['target'])[0]}_cfa_mask.bz2")
    targetsrgb_uint8 = raw_16bit_postprocess(target__cfa_data_patch, target_metadata_dict, target_cfa_mask)
    cv2.imwrite(target_sample_dir, targetsrgb_uint8)