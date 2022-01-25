import os
import glob
import numpy as np
import cv2
import json
import copy
import tqdm
import sys
import random
import time
import multiprocessing as mp
from inspect import getframeinfo, stack
from pathlib import Path

from rawRGB.common.module_raw_utils import *
from rawRGB.common.module_rw import *
from rawRGB.common.module_utils import *


def print_wrap(*args, **kwargs):
    caller = getframeinfo(stack()[1][0])
    print(f"\n:: FN: {caller.filename}, Line: {caller.lineno} \n::", *args, **kwargs)
    time.sleep(0.1)


def multiprocess_with_tqdm(func, work_list):
    print_wrap(f'{os.cpu_count()}')
    pool = mp.Pool(processes=os.cpu_count())
    results = []
    for result in tqdm.tqdm(pool.imap_unordered(func, work_list),
                            total=len(work_list)):
        results.append(result)
    return results


def read_text(filename):
    """
    Read text.
    """
    my_list = []
    if os.path.isfile(filename):
        f = open(filename, 'r')
        lines = f.readlines()
        for line in lines:
            line = line.strip()  # 줄 끝의 줄 바꿈 문자를 제거한다.
            my_list.append(line)
        f.close()
    return my_list


def write_text(filename, new_item):
    """
    Write text if the text is not in the file.
    new_item can be list or not
    """
    my_list = read_text(filename)
    new_item_list = []
    if isinstance(new_item, list):
        new_item_list = new_item
    else:
        new_item_list.append(new_item)

    for new_item_ in new_item_list:
        if not new_item_ in my_list:
            with open(filename, "a") as f:
                f.write(f"{new_item_}\n")



def get_screens(json_dir):

    try:
        with open(json_dir, 'r') as f:
            json_data = json.load(f)

        # print(json_data.keys())
        # print(json_data['Raw_Data_Info.'])
        # print(json_data['Source_Data_Info.'])
        # print(json_data['Learning_Data_Info.'])

        h, w = json_data['Raw_Data_Info.']['Resolution'].split(',')

        w, h = float(w)/3, float(h)/3
        # print('w h :', w, h)

        annotation = json_data['Learning_Data_Info.']['Annotation']
        # print(len(annotation))
        # print(annotation)

        img = np.full((int(w), int(h), 3), 0, dtype=np.uint8)

        drawImg = None

        for i, anno in enumerate(json_data['Learning_Data_Info.']["Annotation"]):

            # class id max is '38'
            json_class_id = anno['Class_ID'][-2:]
            rgb_val = int(json_class_id) * 5

            json_polygon = anno['segmentation']

            if len(json_polygon) % 2 == 0:

                n = 2
                polygon_split_to_2 = [json_polygon[i * n:(i + 1) * n] for i in
                                      range((len(json_polygon) + n - 1) // n)]

                pts = np.array(polygon_split_to_2, dtype=np.int32)

                color = rgb_val

                #for cnt in json_polygon:
                drawImg = cv2.fillPoly(img, [pts], (color, color, color))

                # The labels are build up layer by layer.
                # cv2.imwrite(f'my_json_{str(i).zfill(3)}_{json_class_id}.png', drawImg)
            else:
                return None

        ##########################################################
        # binary json
        # ban (17, 22, 23, 27) * 5
        ban_ID_list = [17, 22, 23, 27]
        ban_rgb_list = [x * 5 for x in ban_ID_list]

        new_map = np.zeros_like(drawImg)
        for item in ban_rgb_list:
            drawImg_c = copy.deepcopy(drawImg)
            drawImg_c[drawImg != item] = 0.0
            drawImg_c[drawImg == item] = 1.0
            new_map += drawImg_c

        drawImg = np.clip(new_map, 0, 1)
        ##########################################################

        return drawImg

    # I don't know what the error reason is but just skip this json.
    except:
        return None


def get_dng_dir_list(RAW_version_list):
    DNG_dir_list = []
    for RAW_version in RAW_version_list:
        DNG_dir_list += glob.glob(f"{RAW_version}/*.dng")
    return DNG_dir_list


class JsonErrorFinder:
    def __init__(self, json_error_txt):
        print_wrap('find_dng_dir_with_json_error')
        self.json_error_txt = json_error_txt

    def find_dng_dir_with_json_error(self, DNG_dir):
        my_json = os.path.splitext(DNG_dir)[0] + '_0001.json'

        if not os.path.isfile(my_json):
            write_text(self.json_error_txt, DNG_dir)
            return DNG_dir
        else:
            drawImg = get_screens(my_json)
            if drawImg is None:
                write_text(self.json_error_txt, DNG_dir)
                return DNG_dir


def print_in_function():
    this_function_name = sys._getframe().f_code.co_name

def get_dng_dir_dict(DNG_dir_list):

    my_dict = {}

    print_wrap('get_dng_dir_dict')
    for db_dir in tqdm.tqdm(DNG_dir_list):
        # db_dir sample : D-210713_I8028L04_007.dng

        # (1) date
        date = os.path.basename(db_dir).split('_')[0].split('-')[1]
        # (2) video_num
        video_num = os.path.basename(db_dir).split('_')[2].split('.')[0]
        # (3) place_id, noise_level
        place_id, noise_level = os.path.basename(db_dir).split('_')[1].split('L')

        # This information is combined to create a key.
        my_key = f"{date}_{place_id}"

        if my_key not in my_dict:
            my_dict[my_key] = {}
        if noise_level not in my_dict[my_key]:
            my_dict[my_key][noise_level] = {}

        my_dict[my_key][noise_level][video_num] = db_dir

    # Show detail infos.
    print_wrap('DNG_dir_dict')
    for my_key in my_dict.keys():
        print(f'key : {my_key}, items : {my_dict[my_key].keys()}')
    return my_dict


def dataset_error_finder(DNG_dir_dict):
    # Find error dataset
    total_dict_error = {}
    for my_key in DNG_dir_dict.keys():
        error = False
        if 'GT' not in DNG_dir_dict[my_key].keys():
            error = True

        if error:
            total_dict_error[my_key] = DNG_dir_dict[my_key]

    # delete error keys from total_dict
    for my_key in total_dict_error.keys():
        DNG_dir_dict.pop(my_key)

    # print error dataset

    print_wrap('Each image Error information (It is excluded from the training data set)')
    for my_key in total_dict_error.keys():
        print(f'key : {my_key}, items : {total_dict_error[my_key].keys()}')

    return DNG_dir_dict, total_dict_error


def left_one_gt_for_noises(DNG_dir_dict):
    # find clean-est image among GT (but not first but second)
    print_wrap('left_one_gt_for_noises, Delete bad GTs')
    for key in tqdm.tqdm(DNG_dir_dict.keys()):
        if len(DNG_dir_dict[key]['GT'].keys()) > 1:

            temp_gt_list = []
            for gt_key in DNG_dir_dict[key]['GT'].keys():
                metadata = extMetadata(DNG_dir_dict[key]['GT'][gt_key])
                temp_gt_list.append([gt_key, metadata['iso']])
            temp_gt_list = sorted(temp_gt_list, key=lambda x: x[1])
            clean_gt_key = temp_gt_list[1][0]  # second clean

            # delete every items except clean gt.
            for gt_key in DNG_dir_dict[key]['GT'].keys():
                if gt_key != clean_gt_key:
                    os.remove(DNG_dir_dict[key]['GT'][gt_key])

            temp = DNG_dir_dict[key]['GT'][clean_gt_key]
            DNG_dir_dict[key]['GT'].clear()
            DNG_dir_dict[key]['GT'][clean_gt_key] = temp

    return DNG_dir_dict


def DNG_dir_dict_to_list(DNG_dir_dict):
    my_lit = []
    for key1 in DNG_dir_dict.keys():
        for key2 in DNG_dir_dict[key1].keys():
            for key3 in DNG_dir_dict[key1][key2].keys():
                my_lit.append(DNG_dir_dict[key1][key2][key3])
    return my_lit


class DNGtoPatches:
    def __init__(self, DB_dir):
        print_wrap('Generate_patches.')
        self.DB_dir = DB_dir

    def generate_patches(self, my_dng_dir):
        # Generate patches from RAW.
        cropped_result_dir = f'{self.DB_dir}_patches/{os.path.basename(Path(my_dng_dir).parent)}'

        # Get the base name.
        dng_base_name = os.path.splitext(os.path.basename(my_dng_dir))[0]

        # Get the 16bit raw image from dng
        img = get_raw_16bit_from_dng(my_dng_dir)

        # Save metadata and dng data.
        metadata = extMetadata(my_dng_dir)

        with rp.imread(my_dng_dir) as raw_obj:
            cfa_mask = raw_obj.raw_colors_visible
            blk_level = raw_obj.black_level_per_channel
            sat_level = raw_obj.white_level
            cfa_type = raw_obj.raw_pattern

        metadata_dict = {'metadata': metadata,
                         'blk_level': blk_level,
                         'sat_level': sat_level,
                         'cfa_type': cfa_type}


        pickle_dir = f'{cropped_result_dir}/{dng_base_name}___metadata_dict'

        # png, pkl, gz, bz2, lzma
        rw_mode = 'bz2'

        write_obj(pickle_dir, metadata_dict, mode=rw_mode)

        # Crop the raw image and save
        ori_w_length = img.shape[1]
        ori_h_length = img.shape[0]

        # Crop size. The size must be even.
        wl = 128 * 3
        hl = 128 * 3
        img = img

        # Crop!
        w_pos = 0
        h_pos = 0
        total_count = 0
        while ori_h_length > h_pos:
            while ori_w_length > w_pos:

                if w_pos + wl > ori_w_length:
                    w_pos = ori_w_length - wl

                if h_pos + hl > ori_h_length:
                    h_pos = ori_h_length - hl

                j, i, w, h = w_pos, h_pos, wl, hl

                # crop image.
                cropped_img = img[i:(i + h), j:(j + w)]
                cropped_cfa_mask = cfa_mask[i:(i + h), j:(j + w)]

                # Save cropped image.
                patch_name = f"{dng_base_name}___{j}_{i}_{w}_{h}"
                write_obj(f'{cropped_result_dir}/{patch_name}', cropped_img, mode=rw_mode)

                # Save cropped cfa_mask.
                patch_name_cfa_mask = f"{dng_base_name}___{j}_{i}_{w}_{h}_cfa_mask"
                write_obj(f'{cropped_result_dir}/{patch_name_cfa_mask}', cropped_cfa_mask, mode=rw_mode)

                total_count += 1

                w_pos += wl

            w_pos = 0
            h_pos += hl

        # os.remove(my_dng_dir)


def check_dict_dirs_isfile(my_dict):
    for my_dict_key in my_dict.keys():
        if not os.path.isfile(my_dict[my_dict_key]):
            raise Exception(f'{my_dict[my_dict_key]} is not exist.')


def DB_dir_to_paired_list(DB_dir, json_folder_dir):
    # Read all the RAW versions.
    RAW_version_list = [os.path.join(DB_dir, x) for x in sorted(os.listdir(DB_dir))]
    # only directories (except wrong folders)
    RAW_version_list = [tempdir for tempdir in RAW_version_list if os.path.isdir(tempdir)]

    print_wrap('real all the .bz2')
    raw_dir_list = []
    for RAW_version in tqdm.tqdm(RAW_version_list):
        raw_dir_list += glob.glob(f"{RAW_version}/*.bz2")

    # Except 'cfa_mask':
    raw_dir_list = [tempdir for tempdir in raw_dir_list if not 'cfa_mask' in tempdir]

    # DNG_dir_list to dict
    my_dict = {}

    print_wrap('get_dng_dir_dict')
    for db_dir in tqdm.tqdm(raw_dir_list):
        # print(db_dir)
        # db_dir sample : D-210712_I6019L01_002___3072_3072_768_768.bz2
        DNG_info, crop_info = os.path.basename(db_dir).split('___')
        crop_info = os.path.splitext(crop_info)[0]

        # (1) date : 210712
        date = DNG_info.split('_')[0].split('-')[1]
        # (2) video_num : 002
        video_num = DNG_info.split('_')[2]
        # (3) place_id, noise_level : I6019, 01
        place_id, noise_level = DNG_info.split('_')[1].split('L')

        # This information is combined to create a key.
        my_key = f"{date}_{place_id}"

        if my_key not in my_dict.keys():
            my_dict[my_key] = {}
        if noise_level not in my_dict[my_key].keys():
            my_dict[my_key][noise_level] = {}
        if video_num not in my_dict[my_key][noise_level].keys():
            my_dict[my_key][noise_level][video_num] = {}


        my_dict[my_key][noise_level][video_num][crop_info] = db_dir

    # dict to list
    print_wrap('dict to list')
    raw_dir_list = []
    for key1 in tqdm.tqdm(my_dict.keys()):
        key2s = list(my_dict[key1].keys())
        key2s.remove('GT')
        GT = list(my_dict[key1]['GT'].values())[0]

        for key2 in key2s:

            for key3 in my_dict[key1][key2].keys():
                key4s = list(my_dict[key1][key2][key3].keys())
                key4s.remove('metadata_dict')
                metadata_dict = my_dict[key1][key2][key3]['metadata_dict']

                for key4 in key4s:

                    input_patch_dir = my_dict[key1][key2][key3][key4]

                    # Get json dir.
                    p = os.path.basename(Path(input_patch_dir).parent)
                    s = os.path.basename(os.path.splitext(input_patch_dir)[0]).split('___')[0]
                    my_json_dir = f'{json_folder_dir}/{p}/{s}_0001.json'
                    if not os.path.isfile(my_json_dir):
                        Exception(f'{my_json_dir} is not exist.')

                    my_item = {
                        'input': my_dict[key1][key2][key3][key4],
                        'target': GT[key4],
                        'json': my_json_dir,
                        'input_metadata_dict': metadata_dict,
                        'target_metadata_dict': GT['metadata_dict']
                    }

                    # check file exist.
                    check_dict_dirs_isfile(my_item)

                    raw_dir_list.append(my_item)

    return raw_dir_list