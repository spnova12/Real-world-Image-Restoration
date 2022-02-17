import rawRGB.nlowlight_indoor.raw_test as raw_test


def test(pretrain_net_dir_for_test, DB_dir, cuda_num=None):
    hf_patches_folder_dir = DB_dir + '_patches'
    json_folder_dir = DB_dir
    raw_test.main2(pretrain_net_dir_for_test, hf_patches_folder_dir, json_folder_dir, cuda_num)

def test_my_db(pretrain_net_dir_for_test, input_folder_dir, out_folder_name, cuda_num=None):
    raw_test.main(pretrain_net_dir_for_test, input_folder_dir, out_folder_name, cuda_num)
    pass
