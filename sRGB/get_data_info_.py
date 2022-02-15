
from sRGB.preprocessing.module_DB_manager import get_human_forrest_db, get_target_noisy_list
from sRGB.preprocessing.module_DB_get_samples import get_DB_high_level_samples


def get_data_info(DB_dir):
    total_dict = get_human_forrest_db(DB_dir, show_details=False, check_json=False)

# you can save some representative sample images.
def get_data_info_and_samples(DB_dir, img_name_to_find=None):

    total_dict = get_human_forrest_db(DB_dir, show_details=False, check_json=False)
    get_DB_high_level_samples(total_dict, img_name_to_find)


