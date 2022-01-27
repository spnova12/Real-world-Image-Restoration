
from sRGB.preprocessing.module_DB_manager import get_human_forrest_db


def get_data_info(DB_dir):
    get_human_forrest_db(DB_dir, show_details=False, check_json=False)
