
# mass includes
import colour
import pyexiv2 as exiv2
from colour_demosaicing import demosaicing_CFA_Bayer_DDFAPD as DDFAPD
import rawpy as rp
import numpy as np
import math
import time
import cv2


def unifyBayerPtn(in_cfa, cfa_type):
    # check bayer type
    color = {0: 'R', 1: 'G', 2: 'B', 3: 'G'}
    cfa_type = cfa_type.reshape(-1)
    cfa_type = color[cfa_type[0]] + color[cfa_type[1]] + color[
        cfa_type[2]] + color[cfa_type[3]]

    # GRBG to RGGB
    if cfa_type == 'GRBG':
        out_cfa = in_cfa[:, 1:-1].copy()

    # GBRG to RGGB
    elif cfa_type == 'GBRG':
        out_cfa = in_cfa[1:-1, :].copy()

    # BGGR to RGGB
    elif cfa_type == 'BGGR':
        out_cfa = in_cfa[1:-1, 1:-1].copy()

    # native RGGB
    else:
        out_cfa = in_cfa.copy()

    return out_cfa


# extract metadata
def extMetadata(file_path):
    # load all metadata
    file_md = exiv2.ImageMetadata(file_path)
    file_md.read()

    # metadata dictionary with default params
    metadata = {}
    metadata['cali_mat1'] = np.identity(3, dtype=np.float32)
    metadata['cali_mat2'] = np.identity(3, dtype=np.float32)
    metadata['ab_mat'] = np.identity(3, dtype=np.float32)
    metadata['cali_flag'] = False

    for key in file_md:
        # extract camera white point
        if 'AsShotNeutral' in key:
            shot_wp = file_md[key].value
            shot_wp = np.array(shot_wp, dtype=np.float32)
            metadata['shot_wb'] = shot_wp
            continue

        # extract calibration illuminations
        if 'CalibrationIlluminant1' in key:
            cali_illum1 = file_md[key].value
            metadata['cali_illum1'] = cali_illum1
            continue
        if 'CalibrationIlluminant2' in key:
            cali_illum2 = file_md[key].value
            metadata['cali_illum2'] = cali_illum2
            continue

        # extract color matrices
        if 'ColorMatrix1' in key:
            clr_mat1 = file_md[key].value
            clr_mat1 = np.array(clr_mat1, dtype=np.float32).reshape((3, 3))
            metadata['clr_mat1'] = clr_mat1
            continue
        if 'ColorMatrix2' in key:
            clr_mat2 = file_md[key].value
            clr_mat2 = np.array(clr_mat2, dtype=np.float32).reshape((3, 3))
            metadata['clr_mat2'] = clr_mat2
            continue

        # extract calibration matrices
        if 'CameraCalibration1' in key:
            cali_mat1 = file_md[key].value
            cali_mat1 = np.array(cali_mat1, dtype=np.float32).reshape((3, 3))
            metadata['cali_mat1'] = cali_mat1
            continue
        if 'CameraCalibration2' in key:
            cali_mat2 = file_md[key].value
            cali_mat2 = np.array(cali_mat2, dtype=np.float32).reshape((3, 3))
            metadata['cali_mat2'] = cali_mat2
            continue

        # extract forward matrices
        if 'ForwardMatrix1' in key:
            fwd_mat1 = file_md[key].value
            fwd_mat1 = np.array(fwd_mat1, dtype=np.float32).reshape((3, 3))
            metadata['fwd_mat1'] = fwd_mat1
            continue
        if 'ForwardMatrix2' in key:
            fwd_mat2 = file_md[key].value
            fwd_mat2 = np.array(fwd_mat2, dtype=np.float32).reshape((3, 3))
            metadata['fwd_mat2'] = fwd_mat2
            metadata['cali_flag'] = True

        # camera ISO
        if 'ISOSpeedRatings' in key:
            cam_iso = file_md[key].value
            metadata['iso'] = cam_iso
            continue

        if 'NoiseProfile' in key:
            cam_noise = file_md[key].raw_value.split()
            cam_noise = np.array(cam_noise, dtype=np.float32)
            metadata['noise'] = cam_noise
            continue

    return metadata



# convert XYZ to color temperature
def color2Mierd(sample):
    # convert to CIE xy chromaticity
    xy = colour.XYZ_to_xy(sample)

    # convert to color temperature
    tcp = colour.xy_to_CCT(xy, method='McCamy 1992')

    # convert to Mierd
    mierd = 1e+6 / tcp

    return mierd


# find asShotNeutural in XYZ space
def cam2xyzWP(metadata, max_iter=100, acc=1e-8):
    # find calibration light source
    illum_list = {
        17: [1.09850, 1.0, 0.35585],
        18: [0.99072, 1.0, 0.85223],
        19: [0.98074, 1.0, 1.18232],
        20: [0.95682, 1.0, 0.92149],
        21: [0.95047, 1.0, 1.08883],
        22: [0.94972, 1.0, 1.22638],
        23: [0.96422, 1.0, 0.82521]
    }
    cali_illum1 = illum_list[metadata['cali_illum1']]
    cali_illum2 = illum_list[metadata['cali_illum2']]
    cali1_tcp = color2Mierd(cali_illum1)
    cali2_tcp = color2Mierd(cali_illum2)

    # analog balance
    ab_mat = metadata['ab_mat']

    # perform camera white point interpolation
    if metadata['cali_flag'] == True:
        upd_wp = illum_list[23]
        interp_w = 0.5
        for index in range(max_iter):
            # get white point temperature
            wp_tcp = color2Mierd(upd_wp)

            # handle out-of-range sample
            if wp_tcp >= cali1_tcp:
                interp_w = 1.0
            elif wp_tcp <= cali2_tcp:
                interp_w = 0.0
            else:
                interp_w = (wp_tcp - cali2_tcp) / (cali1_tcp - cali2_tcp)

            # interpolate transform matrices
            cali_mat = interp_w * metadata['cali_mat1'] + (
                1.0 - interp_w) * metadata['cali_mat2']
            clr_mat = interp_w * metadata['clr_mat1'] + (
                1.0 - interp_w) * metadata['clr_mat2']
            xyz2cam = np.matmul(ab_mat, np.matmul(cali_mat, clr_mat))
            cam2xyz = np.linalg.inv(xyz2cam)
            xyz_wp = np.matmul(cam2xyz,
                               metadata['shot_wb'].reshape(-1, 1)).reshape(-1)
            dist = np.linalg.norm(
                colour.XYZ_to_xy(upd_wp) - colour.XYZ_to_xy(xyz_wp))
            upd_wp = xyz_wp

            # exit loop if converge
            if dist < acc:
                break

    # no enough camera, use default
    else:
        interp_w = 1.0
        cali_mat = metadata['cali_mat1']
        clr_mat = metadata['clr_mat1']
        xyz2cam = np.matmul(ab_mat, np.matmul(cali_mat, clr_mat))
        cam2xyz = np.linalg.inv(xyz2cam)
        upd_wp = np.matmul(cam2xyz, metadata['shot_wb'].reshape(-1,
                                                                1)).reshape(-1)

    return upd_wp, interp_w


# Bradford algorithm
def BradfordAdpt(src_xyz, tgt_xyz):
    cat02 = np.array([[0.7328, 0.4296, -0.1624], [-0.7036, 1.6975, 0.0061],
                      [0.0030, 0.0136, 0.9834]])
    src = np.matmul(cat02, src_xyz.reshape(-1, 1)).reshape(-1)
    pcs = np.matmul(cat02, tgt_xyz.reshape(-1, 1)).reshape(-1)
    t_mat = np.array([[pcs[0] / src[0], 0, 0], [0, pcs[1] / src[1], 0],
                      [0, 0, pcs[2] / src[2]]])
    adpt_mat = np.linalg.lstsq(cat02, np.matmul(t_mat, cat02), rcond=None)[0]

    return adpt_mat


# compute camera to XYZ(D50) transform matrix
def cam2xyzD50(metadata, wp_xyz, interp_w):
    # analog balance
    ab_mat = metadata['ab_mat']

    # interpolate transform matrices
    cali_mat = interp_w * metadata['cali_mat1'] + (
        1 - interp_w) * metadata['cali_mat2']
    clr_mat = interp_w * metadata['clr_mat1'] + (
        1 - interp_w) * metadata['clr_mat2']

    # with forward matrices
    if metadata['cali_flag'] == True:
        # interpolate forward mat
        fwd_mat = interp_w * metadata['fwd_mat1'] + (
            1 - interp_w) * metadata['fwd_mat2']

        # compute camera to XYZ(D50) transform matrix
        abcc_inv = np.linalg.inv(np.matmul(ab_mat, cali_mat))
        ref_wp = np.matmul(abcc_inv, metadata['shot_wb'])
        wb_diag = np.linalg.inv(np.diag(ref_wp.reshape(-1)))
        cam2xyz_d50 = np.matmul(fwd_mat, np.matmul(wb_diag, abcc_inv))

    #without forward matrices
    else:
        d50_xyz = np.array([0.9642, 1.0, 0.8252])
        xyz2cam = np.matmul(ab_mat, np.matmul(cali_mat, clr_mat))
        cam2xyz = np.linalg.inv(xyz2cam)
        ca_dig = BradfordAdpt(wp_xyz, d50_xyz)
        cam2xyz_d50 = np.matmul(ca_dig, cam2xyz)

    return cam2xyz_d50


def demosaic(in_raw, ptn='RGGB'):
    # unfold to flat bayer
    flat_raw = np.zeros((int(in_raw.shape[0] * 2), int(in_raw.shape[1] * 2)))
    flat_raw[0::2, 0::2] = in_raw[:, :, 0].copy()
    flat_raw[0::2, 1::2] = in_raw[:, :, 1].copy()
    flat_raw[1::2, 0::2] = in_raw[:, :, 2].copy()
    flat_raw[1::2, 1::2] = in_raw[:, :, 3].copy()

    # demosaic
    out_img = DDFAPD(flat_raw, pattern=ptn)

    return out_img


def cam2sRGB(raw_img, cam2xyz):
    xyz2srgb = np.array([[3.1339, -1.6169, -0.4907],\
                         [-0.9784, 1.9158, 0.0334],\
                         [0.0720, -0.2290, 1.4057]])
    raw_img = demosaic(raw_img)
    hei, wid, _ = raw_img.shape
    raw_img = np.transpose(raw_img, (2, 0, 1)).reshape(3, -1)
    xyz_img = np.matmul(cam2xyz, raw_img)
    xyz_img = np.clip(xyz_img, 0.0, 1.0)
    srgb_img = np.matmul(xyz2srgb, xyz_img)
    srgb_img = np.clip(srgb_img, 0.0, 1.0)
    srgb_img = srgb_img.reshape((3, hei, wid))
    srgb_img = np.transpose(srgb_img, (1, 2, 0))

    return srgb_img


# gamma correction
def gamma_correction(img, c=1, g=2.2):
    out = img.copy()
    out = (1/c * out) ** (1/g)
    return out


##############################################################################################################

def get_raw_16bit_from_dng(dng_dir):
    with rp.imread(dng_dir) as raw_obj:
        cfa_data = raw_obj.raw_image_visible.copy()

    return cfa_data


def raw_16bit_postprocess(cfa_data, metadata_dict, cfa_mask):
    # compute ISO, noise model, and color matrix
    metadata = metadata_dict['metadata']
    wp_xyz, interp_w = cam2xyzWP(metadata)
    cam2xyz = cam2xyzD50(metadata, wp_xyz, interp_w)

    blk_level = metadata_dict['blk_level']
    sat_level = metadata_dict['sat_level']
    cfa_type = metadata_dict['cfa_type']

    # normalize to 0-1
    cfa_data = cfa_data.astype(np.float32)
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

    srgb_img = cam2sRGB(normalized_raw, cam2xyz)

    # postprocessing
    srgb_img = gamma_correction(srgb_img)
    srgb_img *= 255
    srgb_img = np.around(srgb_img)
    srgb_img = srgb_img.clip(0, 255)
    srgb_img = srgb_img.astype(np.uint8)
    srgb_img = cv2.cvtColor(srgb_img, cv2.COLOR_RGB2BGR)

    return srgb_img


def get_normalized_raw_from_dng(dng_dir):
    with rp.imread(dng_dir) as raw_obj:
        cfa_data = raw_obj.raw_image_visible.copy()
        cfa_mask = raw_obj.raw_colors_visible
        blk_level = raw_obj.black_level_per_channel
        sat_level = raw_obj.white_level
        cfa_type = raw_obj.raw_pattern

    # normalize to 0-1
    cfa_data = cfa_data.astype(np.float32)
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


def normalized_dng_postprocess(normalized_raw, dng_dir):
    # compute ISO, noise model, and color matrix
    metadata = extMetadata(dng_dir)
    wp_xyz, interp_w = cam2xyzWP(metadata)
    cam2xyz = cam2xyzD50(metadata, wp_xyz, interp_w)
    srgb_img = cam2sRGB(normalized_raw, cam2xyz)

    # postprocessing
    srgb_img = gamma_correction(srgb_img)
    srgb_img *= 255
    srgb_img = np.around(srgb_img)
    srgb_img = srgb_img.clip(0, 255)
    srgb_img = srgb_img.astype(np.uint8)
    srgb_img = cv2.cvtColor(srgb_img, cv2.COLOR_RGB2BGR)

    return srgb_img


def normalized_dng_postprocess_for_patch(normalized_raw, metadata_dict):
    # compute ISO, noise model, and color matrix
    metadata = metadata_dict['metadata']
    wp_xyz, interp_w = cam2xyzWP(metadata)
    cam2xyz = cam2xyzD50(metadata, wp_xyz, interp_w)
    srgb_img = cam2sRGB(normalized_raw, cam2xyz)

    # postprocessing
    srgb_img = gamma_correction(srgb_img)
    srgb_img *= 255
    srgb_img = np.around(srgb_img)
    srgb_img = srgb_img.clip(0, 255)
    srgb_img = srgb_img.astype(np.uint8)
    srgb_img = cv2.cvtColor(srgb_img, cv2.COLOR_RGB2BGR)

    return srgb_img



def raw_16bit_to_normalized_raw(raw_16bit, metadata_dict, cfa_mask):
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
