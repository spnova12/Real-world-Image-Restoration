import copy
import random
import json
import os
import numpy as np
import cv2
import tqdm
from PIL import Image


def numpyPSNR(tar_img_dir, prd_img_dir):

    tar_img = cv2.imread(tar_img_dir)
    prd_img = cv2.imread(prd_img_dir)

    imdff = np.float32(prd_img) - np.float32(tar_img)
    rmse = np.sqrt(np.mean(imdff**2))
    ps = 20*np.log10(255/rmse)
    return ps


def median_imgs(img_dirs):

    imgs_r = []
    imgs_g = []
    imgs_b = []

    for img_dir in img_dirs:
        img = cv2.imread(img_dir)
        imgs_b.append(img[:, :, 0])
        imgs_g.append(img[:, :, 1])
        imgs_r.append(img[:, :, 2])

    imgs_r = np.median(np.stack(imgs_r, axis=0), axis=0)
    imgs_g = np.median(np.stack(imgs_g, axis=0), axis=0)
    imgs_b = np.median(np.stack(imgs_b, axis=0), axis=0)

    imgs = np.stack([imgs_b, imgs_g, imgs_r], axis=2)
    return imgs


class MedianImgs():
    def __init__(self, img_dirs):
        self.imgs_list = []

        imgs_r = []
        imgs_g = []
        imgs_b = []

        for img_dir in img_dirs:
            img = cv2.imread(img_dir)
            self.imgs_list.append(img)
            imgs_b.append(img[:, :, 0])
            imgs_g.append(img[:, :, 1])
            imgs_r.append(img[:, :, 2])

        imgs_r = np.median(np.stack(imgs_r, axis=0), axis=0)
        imgs_g = np.median(np.stack(imgs_g, axis=0), axis=0)
        imgs_b = np.median(np.stack(imgs_b, axis=0), axis=0)

        self.median_img = np.stack([imgs_b, imgs_g, imgs_r], axis=2)

    def get_median_result(self):
        return self.median_img

    def get_imgs_list(self):
        return self.imgs_list



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

    # Delete duplicates.
    my_list = list(set(my_list))
    return my_list


def write_text(filename, new_item, valid_path_start_idx):
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

        new_item_ = '/'.join(new_item_.split('/')[valid_path_start_idx:])

        if not new_item_ in my_list:
            with open(filename, "a") as f:
                f.write(f"{new_item_}\n")

def is_dir_in_list(dir, list, valid_path_start_idx):
    my_item = '/'.join(dir.split('/')[valid_path_start_idx:])
    if my_item in list:
        return True
    else:
        return False


def get_human_forrest_db(DB_dir, show_details=False, check_json=False):
    ############################################################################################################
    # Figure out Each version, read meta infos (error txt), init
    ############################################################################################################
    R_F_D_S_C = [os.path.join(DB_dir, x) for x in sorted(os.listdir(DB_dir))]

    # only directories (except wrong folders)
    R_F_D_S_C = [tempdir for tempdir in R_F_D_S_C if os.path.isdir(tempdir)]

    # todo : for toy exp.
    R_F_D_S_C = R_F_D_S_C  # [:2]

    print(f"\n:: sRGB DB list : \n{[os.path.basename(bname) for bname in R_F_D_S_C]}")

    # Get this script's dir.
    preprocessing_dir = os.path.dirname(os.path.realpath(__file__))

    # load error json list npy
    error_json_list_dir = f'{preprocessing_dir}/error_json_list.txt'
    error_json_list = read_text(error_json_list_dir)

    error_size_list_dir = f'{preprocessing_dir}/error_size_list.txt'
    error_size_list = read_text(error_size_list_dir)

    # init my_dict_per_version
    my_dict_per_version = []

    # init images_len_sum
    images_len_sum = 0

    # If the DB's json is checked than skip reading this.
    R_F_D_S_C_checked_txt = f"{preprocessing_dir}/R_F_D_S_C_checked.txt"
    R_F_D_S_C_list = []
    if os.path.isfile(R_F_D_S_C_checked_txt):
        R_F_D_S_C_list = read_text(R_F_D_S_C_checked_txt)

    ############################################################################################################
    # Read all noises one by one, Check json Errors.
    ############################################################################################################
    for version_dir in R_F_D_S_C:
        error_json_list_new = []

        version_base_name = os.path.basename(version_dir)

        is_version_in_R_F_D_S_C_list = False
        if is_dir_in_list(version_dir, R_F_D_S_C_list, -1):
            is_version_in_R_F_D_S_C_list = True

        if check_json and is_version_in_R_F_D_S_C_list:
            print(f'\n:: {version_base_name} has already been verified')
            lets_check_db = False
        else:
            lets_check_db = True

        if lets_check_db:
            DB_list = [os.path.join(version_dir, x) for x in sorted(os.listdir(version_dir))]

            print(f'\n:: Check {version_base_name}. (Initial images count : {len(DB_list)})')
            images_len_sum += len(DB_list)

            noises = ['R', 'F', 'D', 'S', 'L']  # Rain, Fog, Dust, Snow, Lowlight

            my_dict = {}

            # todo :
            DB_list = DB_list
            for db_dir in tqdm.tqdm(DB_list):
                have_error = False

                if is_dir_in_list(db_dir, error_json_list, -2):
                    have_error = True

                if check_json and not have_error:
                    my_json = os.path.splitext(db_dir)[0] + '.json'
                    drawImg = get_sky(my_json)
                    if drawImg is None:
                        error_json_list_new.append(db_dir)
                        have_error = True

                if not have_error:
                    # read only image not json.
                    my_format = os.path.splitext(db_dir)[-1]

                    if my_format in ['.jpg', '.png']:

                        # check image size.
                        size_error = False
                        if not is_dir_in_list(db_dir, error_size_list, -2) and not is_version_in_R_F_D_S_C_list:
                            img_temp = Image.open(db_dir)
                            w, h = img_temp.size
                            if w != 1920 or h != 1080:
                                write_text(error_size_list_dir, db_dir, -2)
                                size_error = True

                        if not size_error:
                            # (1) date
                            date = os.path.basename(db_dir).split('_')[0].split('-')[1]
                            # (2) video_num
                            video_num = os.path.basename(db_dir).split('_')[2]
                            # (3)Noise_id
                            noise_id = None

                            info = os.path.basename(db_dir).split('_')[1]
                            for n in noises:
                                if n in info:
                                    noise_id = n
                                    break

                            if noise_id is not None:
                                # (4) place_id
                                place_id = info.split(noise_id)[0]
                                # (5) noise_level
                                noise_level = info.split(noise_id)[1]

                                # This information is combined to create a key.
                                my_key = f"{date}_{place_id}_{video_num}_{noise_id}"

                                if my_key not in my_dict:
                                    my_dict[my_key] = {}
                                    my_dict[my_key][noise_id] = {'01': [], '02': [], '03': [], '04': [], '05': [],
                                                                 '06': [], 'GT': []}
                                    my_dict[my_key][noise_id][noise_level].append(db_dir)
                                else:
                                    my_dict[my_key][noise_id][noise_level].append(db_dir)

            # print info
            dict_counter = get_count_for_each_noise(my_dict)
            print(':: Number of key values of noise included in this folder.')
            for my_dict_key2 in dict_counter.keys():
                print(f'::   {my_dict_key2} : {len(dict_counter[my_dict_key2])}')

            my_dict_per_version.append(my_dict)

            # backup. (save error json list)
            write_text(error_json_list_dir, error_json_list_new, -2)
            print(f':: Each json Error information : {len(error_json_list_new)} '
                  f'(It is excluded from the training data set. check error_json_list.txt)')

        # back up.
        if lets_check_db:
            write_text(R_F_D_S_C_checked_txt, version_dir, -1)




    ############################################################################################################
    # Merge each version's dict, Show total info.
    ############################################################################################################
    total_dict = {}
    for my_dict_ in my_dict_per_version:
        total_dict.update(my_dict_)

    total_dict_old = copy.deepcopy(total_dict)


    ############################################################################################################
    # Json errors.
    ############################################################################################################


    ############################################################################################################
    # Dataset errors.
    ############################################################################################################
    # Find error dataset
    total_dict_error = {}
    for my_key in total_dict.keys():
        error = False

        for my_key2 in total_dict[my_key].keys():
            if total_dict[my_key][my_key2]:
                len_sum = 0
                for my_key3 in total_dict[my_key][my_key2].keys():

                    # Exclude if gt is not exist.
                    if len(total_dict[my_key][my_key2]['GT']) == 0:
                        error = True

                    #
                    if my_key3 != 'GT':
                        len_sum += len(total_dict[my_key][my_key2][my_key3])

                if len_sum < 50:
                    error = True

        if error:
            total_dict_error[my_key] = total_dict[my_key]

    # delete error keys from total_dict
    for my_key in total_dict_error.keys():
        total_dict.pop(my_key)


    # print error dataset
    print('\n:: Each image Error information (It is excluded from the training data set)')
    for my_key in total_dict_error.keys():
        print('---------------------------')
        print('p_id :', my_key)
        for my_key2 in total_dict_error[my_key].keys():
            if total_dict_error[my_key][my_key2]:
                print('   ', my_key2)
                for my_key3 in total_dict_error[my_key][my_key2].keys():
                    print('      ', my_key3, ':', len(total_dict_error[my_key][my_key2][my_key3]))
                    if my_key3 == 'GT':
                        print('       GT dir :', total_dict_error[my_key][my_key2][my_key3])



    print('\n:: total size')
    print(':: images len :', images_len_sum)
    dict_counter = get_count_for_each_noise(total_dict_old)
    for my_dict_key2 in dict_counter.keys():
        print(f':: {my_dict_key2} : {len(dict_counter[my_dict_key2])}')

    print('\n:: total size (After error removal)')
    dict_counter = get_count_for_each_noise(total_dict)
    for my_dict_key2 in dict_counter.keys():
        print(f':: {my_dict_key2} : {len(dict_counter[my_dict_key2])}')


    ############################################################################################################
    # Option. Show detail infos.
    ############################################################################################################
    if show_details:
        print('\n:: Each image information')
        for my_key in total_dict.keys():
            print('---------------------------')
            print('p_id :', my_key)
            for my_key2 in total_dict[my_key].keys():
                if total_dict[my_key][my_key2]:
                    print('   ', my_key2)
                    for my_key3 in total_dict[my_key][my_key2].keys():
                        print('      ', my_key3, ':', len(total_dict[my_key][my_key2][my_key3]))
                        if my_key3 == 'GT':
                            print('       GT dir :', total_dict[my_key][my_key2][my_key3])

    return total_dict


def get_target_noisy_list(total_dict, target_noise_type):
    ############################################################################################################
    # Return the target Dataset.
    ###########################################################################################################
    my_DB = []
    for my_key in total_dict.keys():
        for my_key2 in total_dict[my_key].keys():
            if total_dict[my_key][my_key2]:
                if my_key2 == target_noise_type:
                    my_DB.append(total_dict[my_key][my_key2])
    return my_DB


def get_count_for_each_noise(my_dict):
    dict_counter = {}
    for my_dict_key in my_dict.keys():
        for my_dict_key2 in my_dict[my_dict_key].keys():
            if my_dict_key2 not in dict_counter:
                dict_counter[my_dict_key2] = [my_dict_key]
            else:
                dict_counter[my_dict_key2].append(my_dict_key)

    return dict_counter


class HumanForrestManager:
    def __init__(self, DB_dir, target_noise_type, show_details=False, check_json=False):
        # ['R', 'F', 'D', 'S', 'L']
        # Example.
        # get L DB
        total_dict = get_human_forrest_db(DB_dir, show_details, check_json)
        self.my_db = get_target_noisy_list(total_dict, target_noise_type)

    def get_db_len(self):
        return len(self.my_db)

    def get_input_target_pairs(self, my_idx, noise_level=None, noisy_num=None, median=False):
        # The median is taken from the images with a margin around the center image.
        margin = 3

        if noise_level is None:
            levels = list(self.my_db[my_idx].keys())
            levels.remove('GT')

            # If the level has no items then delete from levels.
            levels = [l for l in levels if len(self.my_db[my_idx][l]) > (margin * 2 + 1)]

            level = random.choice(levels)
        else:
            if type(noise_level) == int:
                noise_level = f'0{str(noise_level)}'
            level = noise_level

        if not self.my_db[my_idx][level]:
            # print('------------------------------')
            # print('Error idx :', my_idx)
            # print(self.my_db[my_idx][level])
            # print(self.my_db[my_idx])
            # print('==============================')
            # quit()
            return None
        else:
            if median:
                if noisy_num == None:
                    noisy_num = random.randint(margin, len(self.my_db[my_idx][level])-margin-1)
                    input = []
                    for i in range(margin * 2 + 1):
                        input.append(self.my_db[my_idx][level][noisy_num - margin + i])
                else:
                    input = []
                    for i in range(margin * 2 + 1):
                        input.append(self.my_db[my_idx][level][noisy_num - margin + i])
            else:
                if noisy_num == None:
                    input = random.choice(self.my_db[my_idx][level])
                else:
                    input = self.my_db[my_idx][level][noisy_num]

        target = self.my_db[my_idx]['GT'][0]

        return input, target


def get_sky(json_dir):
    with open(json_dir, 'r') as f:
        json_data = json.load(f)

    # print(json_data.keys())
    # print(json_data['Raw_Data_Info.'])
    # print(json_data['Source_Data_Info.'])
    # print(json_data['Learning_Data_Info.'])

    h, w = json_data['Raw_Data_Info.']['Resolution'].split(',')
    w, h = int(w), int(h)
    # print('w h :', w, h)

    annotation = json_data['Learning_Data_Info.']['Annotation']
    # print(len(annotation))
    # print(annotation)

    img = np.full((w, h, 3), 0, dtype=np.uint8)

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

    # sky's rgb value is '10'
    # labels to sky and the others.
    if drawImg is not None:
        drawImg[drawImg != 10] = 0.0
        drawImg[drawImg == 10] = 1.0

    return drawImg


if __name__=="__main__":
    a = read_text('error_json_list_backup.txt')

    a = ['/'.join(b.split('/')[-2:]) for b in a]
    write_text('error_json_list.txt', a, 0)
    print(a)
