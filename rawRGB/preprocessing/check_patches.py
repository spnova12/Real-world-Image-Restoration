import os
import glob
from rawRGB.common.module_init_raw_DB_utils import *
import argparse

parser = argparse.ArgumentParser(description='check_patches')
parser.add_argument('--DNGs_dir', default='/hdd1/works/datasets/ssd2/human_and_forest/RAW', type=str)
parser.add_argument('--Patches_dir', default='/hdd1/works/datasets/ssd2/human_and_forest/RAW_patches', type=str)
args = parser.parse_args()

json_folder_dir = args.DNGs_dir
DB_dir = args.Patches_dir

# Get the paired list from DB_dir.
raw_dir_paired_list = DB_dir_to_paired_list(DB_dir, json_folder_dir)


# Show the list info.
print_wrap('Show the list info.')
print('len :', len(raw_dir_paired_list))
print('raw_dir_list[0] :', raw_dir_paired_list[0])


# visualize .bz2 sample(16bit raw) to sRGB png.
sample1 = raw_dir_paired_list[0]


# input
print_wrap('Write input')
input_cfa_data_patch = read_obj(sample1['input'])
input_metadata_dict = read_obj(sample1['input_metadata_dict'])
input_cfa_mask = read_obj(f"{os.path.splitext(sample1['input'])[0]}_cfa_mask.bz2")
input_srgb_uint8 = raw_16bit_postprocess(input_cfa_data_patch, input_metadata_dict, input_cfa_mask)
cv2.imwrite('input_patch_sample_sRGB_uint8.png', input_srgb_uint8)


# target
print_wrap('Write target')
target__cfa_data_patch = read_obj(sample1['target'])
target_metadata_dict = read_obj(sample1['target_metadata_dict'])
target_cfa_mask = read_obj(f"{os.path.splitext(sample1['target'])[0]}_cfa_mask.bz2")
targetsrgb_uint8 = raw_16bit_postprocess(target__cfa_data_patch, target_metadata_dict, target_cfa_mask)
cv2.imwrite('target_patch_sample_sRGB_uint8.png', targetsrgb_uint8)






