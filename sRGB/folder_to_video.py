import cv2
import numpy as np
import glob
import tqdm
import copy
import os


def color_median(img_dirs):
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


def folder_to_video(folder_dir, folder_recon_dir, videoname='MyVedio.avi'):

    img_dirs = [os.path.join(folder_dir, x) for x in sorted(os.listdir(folder_dir))]
    img_recon_dirs = [os.path.join(folder_recon_dir, x) for x in sorted(os.listdir(folder_recon_dir))]

    # Set margin
    margin = 2


    # Get img array
    img_array = []

    for filename in img_dirs[margin:-margin]:
        img = cv2.imread(filename)
        img_array.append(img)

    # Get img_recon array
    img_recon_array = []

    for i in tqdm.tqdm(range(margin, len(img_recon_dirs)-margin)):
        median_img = color_median(img_recon_dirs[i-margin:i+margin])
        img_recon_array.append(median_img)


    # Get img size
    height, width, layers = img_array[0].shape
    size = (width, height)


    # Get red for center line
    black = np.zeros((height, width))
    white = np.ones((height, width)) * 255
    red = np.stack((black, black, white), axis=-1)

    # Set start point
    start_point = 17
    img_array = img_array[start_point:]
    img_recon_array = img_recon_array[start_point:]


    # Merge image
    img_array_result = []
    for img, img_recon in tqdm.tqdm(zip(img_array, img_recon_array)):
        img[:, int(width/2):] = img_recon[:, int(width/2):]
        img[:, int(width/2)] = red[:, int(width/2)]

        img_array_result.append(img)


    # Write video
    out = cv2.VideoWriter(f'test-out/{videoname}', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    img_array_result = img_array_result[17:]

    for i in range(len(img_array_result)):
        out.write(img_array_result[i])
    out.release()


def folder_to_video_sliding(folder_dir, folder_recon_dir, videoname='MyVedio_sliding.avi'):
    img_dirs = [os.path.join(folder_dir, x) for x in sorted(os.listdir(folder_dir))]
    img_recon_dirs = [os.path.join(folder_recon_dir, x) for x in sorted(os.listdir(folder_recon_dir))]

    # Set margin. (For video use frame information. So use median)
    margin = 2

    # Get img array.
    img_array = []
    for filename in img_dirs[margin:-margin]:
        img = cv2.imread(filename)
        img_array.append(img)

    # Get img_recon array.
    img_recon_array = []
    for i in tqdm.tqdm(range(margin, len(img_recon_dirs) - margin)):
        median_img = color_median(img_recon_dirs[i - margin:i + margin])
        img_recon_array.append(median_img)

    # Get img size
    height, width, layers = img_array[0].shape
    size = (width, height)

    # Get red for center line
    black = np.zeros((height, width))
    white = np.ones((height, width)) * 255
    red = np.stack((black, black, white), axis=-1)

    # Set start point. (The time for only showing input image)
    start_point = 17

    # if the frames are too small then multiple images.
    img_array = img_array[start_point:] * 2
    img_recon_array = img_recon_array[start_point:] * 2

    # Blending image by sliding.
    img_array_result = []
    denoising_start_frame = 5
    intervel = width / (len(img_array) - denoising_start_frame)

    for i, (img, img_recon) in tqdm.tqdm(enumerate(zip(img_array, img_recon_array))):

        img1 = copy.deepcopy(img)

        if i > denoising_start_frame:
            line_pos = int(intervel * (i - denoising_start_frame + 1))
            if line_pos >= width:
                line_pos = 0

            img1[:, -line_pos:] = img_recon[:, -line_pos:]
            img1[:, -line_pos] = red[:, -line_pos]

        img_array_result.append(copy.deepcopy(img1))

    print('result len', len(img_array_result))

    # Write video
    out = cv2.VideoWriter(f'test-out/{videoname}', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array_result)):
        out.write(img_array_result[i])
    out.release()