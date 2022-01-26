
from sRGB.preprocessing.module_DB_manager import get_human_forrest_db


def preprocessing(DB_dir):
    hf_DB_dict = get_human_forrest_db(DB_dir, show_details=True, check_json=True)



