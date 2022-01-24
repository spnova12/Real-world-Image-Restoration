from rawRGB.common.module_init_raw_DB_utils import *
from rawRGB.common.module_utils import *
import argparse


def init_raw_DB(args):
    DB_dir = args.DNG_dir

    # Read all the RAW versions.
    RAW_version_list = [os.path.join(DB_dir, x) for x in sorted(os.listdir(DB_dir))]
    # only directories (except wrong folders)
    RAW_version_list = [tempdir for tempdir in RAW_version_list if os.path.isdir(tempdir)]


    # mkdir for RAW_patches
    for RAW_version in RAW_version_list:
        make_dirs(f'{DB_dir}_patches/{os.path.basename(RAW_version)}')


    # Exclude what has already checked.
    checked_txt = 'checked.txt'
    checked_RAW_version_list = read_text(checked_txt)
    RAW_version_list = [tempdir for tempdir in RAW_version_list if tempdir not in checked_RAW_version_list]


    # Get all the DNG dirs to list.
    DNG_dir_list = get_dng_dir_list(RAW_version_list)


    # Exclude loaded json error.
    json_error_txt = 'error_report_json.txt'
    DNG_dir_list_with_json_error = read_text(json_error_txt)
    DNG_dir_list = [tempdir for tempdir in DNG_dir_list if tempdir not in DNG_dir_list_with_json_error]


    # Find and save new json error.
    json_error_finder = JsonErrorFinder(json_error_txt)
    DNG_dir_list_with_json_error = multiprocess_with_tqdm(json_error_finder.find_dng_dir_with_json_error, DNG_dir_list)
    # Delete None in the DNG_dir_list_with_json_error.
    DNG_dir_list_with_json_error = list(filter(None.__ne__, DNG_dir_list_with_json_error))
    # Exclude new json error from DNG_dir_list.
    DNG_dir_list = [x for x in DNG_dir_list if x not in DNG_dir_list_with_json_error]


    # Delete DNG with json error


    # Make dict from DNG_dir_list
    DNG_dir_dict = get_dng_dir_dict(DNG_dir_list)


    # Find dataset error and exclude error.
    DNG_dir_dict, total_dict_error = dataset_error_finder(DNG_dir_dict)


    # Delete DNG with error


    # Left one GT for noises.
    DNG_dir_dict = left_one_gt_for_noises(DNG_dir_dict)


    # DNG to List
    DNG_dir_list_refined = DNG_dir_dict_to_list(DNG_dir_dict)


    # DNG to patches.
    dng_to_patches = DNGtoPatches(DB_dir)
    multiprocess_with_tqdm(dng_to_patches.generate_patches, DNG_dir_list_refined)


    # Write RAW_version_list after it checked.
    write_text(checked_txt, RAW_version_list)



if __name__ == '__main__':
    print(os.path.realpath(__file__))
    quit()
    parser = argparse.ArgumentParser(description='Init RAW DB')
    parser.add_argument('--DNGs_dir', default='/hdd1/works/datasets/ssd2/human_and_forest/RAW', type=str)
    args = parser.parse_args()

    init_raw_DB(args)