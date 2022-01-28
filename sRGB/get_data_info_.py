
from sRGB.preprocessing.module_DB_manager import get_human_forrest_db, get_target_noisy_list


def get_data_info(DB_dir):
    total_dict = get_human_forrest_db(DB_dir, show_details=False, check_json=False)



