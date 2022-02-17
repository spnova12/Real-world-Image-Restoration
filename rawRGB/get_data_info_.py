import os
import glob
from rawRGB.common.module_init_raw_DB_utils import *
import argparse
import shutil

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


def get_data_info_and_samples(DB_dir):

    patch_DB_dir = DB_dir + '_patches'
    json_folder_dir = DB_dir

    # Get the paired list from DB_dir.
    raw_dir_paired_list = DB_dir_to_paired_list(patch_DB_dir, json_folder_dir)

    # Show the list info.
    print_wrap('Show the list info.')
    print('len(raw_dir_paired_list) :', len(raw_dir_paired_list))

    out_dir = make_dirs('test-out/DB_raw_samples')

    def save_sample(sample1):
        # input
        bname = os.path.basename(sample1['input'])
        bname = os.path.splitext(bname)[0]

        out_sample_dir = f'{out_dir}/{bname}_input_sRGB_uint8.png'

        cfa_mask_dir = f"{os.path.splitext(sample1['input'])[0]}_cfa_mask.bz2"

        input_cfa_data_patch = read_obj(sample1['input'])
        input_metadata_dict = read_obj(sample1['input_metadata_dict'])
        input_cfa_mask = read_obj(cfa_mask_dir)
        input_srgb_uint8 = raw_16bit_postprocess(input_cfa_data_patch, input_metadata_dict, input_cfa_mask)
        cv2.imwrite(out_sample_dir, input_srgb_uint8)
        shutil.copy(sample1['input'], f'{out_dir}/{bname}.bz2')
        shutil.copy(sample1['input'], f'{out_dir}/{bname}.bz2')

        bname_metadata = os.path.basename(sample1['input_metadata_dict'])
        shutil.copy(sample1['input_metadata_dict'], f'{out_dir}/{bname_metadata}')
        bname_cfa_mask = os.path.basename(cfa_mask_dir)
        shutil.copy(cfa_mask_dir, f'{out_dir}/{bname_cfa_mask}')


        # target
        target_sample_dir = f'{out_dir}/{bname}_target_sRGB_uint8.png'
        # print_wrap(f'Write target sample. ({target_sample_dir})')

        target__cfa_data_patch = read_obj(sample1['target'])
        target_metadata_dict = read_obj(sample1['target_metadata_dict'])
        target_cfa_mask = read_obj(f"{os.path.splitext(sample1['target'])[0]}_cfa_mask.bz2")
        targetsrgb_uint8 = raw_16bit_postprocess(target__cfa_data_patch, target_metadata_dict, target_cfa_mask)
        cv2.imwrite(target_sample_dir, targetsrgb_uint8)


    # sample dataset.
    samples_count = 20
    db_len = len(raw_dir_paired_list)
    interval = db_len // samples_count
    samples_set = []
    for idx in tqdm.tqdm(range(samples_count)):
        input_target_pairs = raw_dir_paired_list[idx * interval]
        samples_set.append(input_target_pairs)

    # visualize .bz2 samples(16bit raw) to sRGB png.
    for sample in samples_set:
        save_sample(sample)