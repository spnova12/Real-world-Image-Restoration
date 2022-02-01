
from sRGB.preprocessing.module_DB_manager import get_human_forrest_db
from sRGB.preprocessing.noisy_gt_aligner_train import noisy_gt_aligner_train

def preprocessing(DB_dir):
    get_human_forrest_db(DB_dir, show_details=False, check_json=True)

def train_align_net(exp_name, DB_dir, noise_type, cuda_num=None):
    noisy_gt_aligner_train(exp_name, DB_dir, noise_type, cuda_num)


