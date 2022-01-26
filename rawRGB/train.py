import rawRGB.nlowlight_indoor.raw_train as raw_train

def train(exp_name, DB_dir):
    hf_patches_folder_dir = DB_dir + '_patches'
    json_folder_dir = DB_dir
    raw_train.main(exp_name, hf_patches_folder_dir, json_folder_dir)