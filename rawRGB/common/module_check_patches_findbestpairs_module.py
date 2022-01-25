from rawRGB.common.module_init_raw_DB_utils import *

def my_raw_16bit_to_normalized_raw(raw_16bit, metadata_dict, cfa_mask):
    # compute ISO, noise model, and color matrix
    blk_level = metadata_dict['blk_level']
    sat_level = metadata_dict['sat_level']
    cfa_type = metadata_dict['cfa_type']

    # normalize to 0-1
    cfa_data = raw_16bit.astype(np.float32)
    cfa_data[cfa_mask == 0] = cfa_data[cfa_mask == 0] - blk_level[0]
    cfa_data[cfa_mask == 1] = cfa_data[cfa_mask == 1] - blk_level[1]
    cfa_data[cfa_mask == 2] = cfa_data[cfa_mask == 2] - blk_level[2]
    cfa_data[cfa_mask == 3] = cfa_data[cfa_mask == 3] - blk_level[3]
    cfa_data = cfa_data / (sat_level - max(blk_level))
    cfa_data = np.clip(cfa_data, 0.0, 1.0)

    # Bayer pattern unification
    cfa_data = unifyBayerPtn(cfa_data, cfa_type)

    # pack to 4-channel raw
    normalized_raw = np.zeros(
        (math.ceil(cfa_data.shape[0] / 2),
         math.ceil(cfa_data.shape[1] / 2), 4))
    normalized_raw[:, :, 0] = cfa_data[0::2, 0::2]
    normalized_raw[:, :, 1] = cfa_data[0::2, 1::2]
    normalized_raw[:, :, 2] = cfa_data[1::2, 0::2]
    normalized_raw[:, :, 3] = cfa_data[1::2, 1::2]

    return normalized_raw


def bz2_to_normalized_raw(bz2_dir, input_metadata_dict):
    input_cfa_mask = read_obj(f"{os.path.splitext(bz2_dir)[0]}_cfa_mask.bz2")
    my_img = my_raw_16bit_to_normalized_raw(read_obj(bz2_dir), read_obj(input_metadata_dict), input_cfa_mask)
    return my_img

def write_normalized_raw(filename, np_img):
    np_img = np_img[:, :, :3]
    np_img = np_img * 255
    cv2.imwrite(filename, np_img)


class FindGoodNoisyPair:
    def __init__(self, gt_based_dict, sample_save_dir=None, real_mode=False):
        self.gt_based_dict = gt_based_dict
        self.sample_save_dir = sample_save_dir

        # if real_mode == True -> delete noisy image, do not write sRGB
        # if real_mode == False -> write sRGB
        self.real_mode = real_mode

    def find_good_noisy_pair(self, gt_key):
        # get gt image
        gt_metadata_dict = self.gt_based_dict[gt_key][0]['target_metadata_dict']
        gt_img = bz2_to_normalized_raw(gt_key, gt_metadata_dict)

        if not self.real_mode:
            # write gt as 8bit sRGB
            input_cfa_data_patch = read_obj(gt_key)
            input_metadata_dict = read_obj(gt_metadata_dict)
            input_cfa_mask = read_obj(f"{os.path.splitext(gt_key)[0]}_cfa_mask.bz2")
            input_srgb_uint8 = raw_16bit_postprocess(input_cfa_data_patch, input_metadata_dict, input_cfa_mask)
            cv2.imwrite(f'{self.sample_save_dir}/GT.png', input_srgb_uint8)

        # get noisy images and cancluate psnr.
        noisy_with_psnr_list = []
        for noisy in self.gt_based_dict[gt_key]:
            noisy_dir = noisy['input']
            noisy_metadata_dict = noisy['input_metadata_dict']
            noisy_img = bz2_to_normalized_raw(noisy_dir, noisy_metadata_dict)

            # get iso.
            my_noisy_metadata_dict = read_obj(noisy_metadata_dict)
            noisy_iso = my_noisy_metadata_dict['metadata']['iso']

            # get psnr.
            psnr = get_psnr(gt_img, noisy_img, max_value=1)
            noisy_with_psnr_list.append([noisy, psnr, noisy_iso])

        ########################################################################################################

        # sort by psnr.
        noisy_with_psnr_list = sorted(noisy_with_psnr_list, key=lambda x: x[1], reverse=True)
        noisy_with_psnr_list_origin = noisy_with_psnr_list.copy()

        # get top 50 imgs
        noisy_with_psnr_list = noisy_with_psnr_list[:50]

        # sort by iso.
        noisy_with_psnr_list = sorted(noisy_with_psnr_list, key=lambda x: x[2], reverse=True)
        # get top 5 imgs
        noisy_with_psnr_list = noisy_with_psnr_list[:5]

        # write noisy images by sorted psnr.
        if not self.real_mode:
            for i, noisy_with_psnr in enumerate(noisy_with_psnr_list):
                # print(i, noisy_with_psnr[0]['input'], noisy_with_psnr[1])
                input_cfa_data_patch = read_obj(noisy_with_psnr[0]['input'])
                input_metadata_dict = read_obj(noisy_with_psnr[0]['input_metadata_dict'])
                input_cfa_mask = read_obj(f"{os.path.splitext(noisy_with_psnr[0]['input'])[0]}_cfa_mask.bz2")
                input_srgb_uint8 = raw_16bit_postprocess(input_cfa_data_patch, input_metadata_dict, input_cfa_mask)
                input_iso = input_metadata_dict['metadata']['iso']
                cv2.imwrite(f'{self.sample_save_dir}/'
                            f'{str(i).zfill(4)}_{str(input_iso).zfill(5)}_{round(noisy_with_psnr[1], 3)}.png',
                            input_srgb_uint8)
        else:
            # delete unselected noisy images.
            list_for_delete = [item[0]['input'] for item in noisy_with_psnr_list_origin if item not in noisy_with_psnr_list]
            for l in list_for_delete:

                dirname = os.path.dirname(l)
                basename = os.path.splitext(os.path.basename(l))[0]
                format = os.path.splitext(os.path.basename(l))[1]
                cfa_mask_dir = f"{dirname}/{basename}_cfa_mask{format}"

                os.remove(l)
                os.remove(cfa_mask_dir)




