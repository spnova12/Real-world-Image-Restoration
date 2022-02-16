import sRGB.R_S_L_F_D.R_S_L_F_D_test as R_S_L_F_D_test
import os

def test_my_db(pretrain_net_dir_for_test, input_folder_dir, out_folder_name, cuda_num):

    R_S_L_F_D_test.main(pretrain_net_dir_for_test, input_folder_dir, out_folder_name, cuda_num)


def test(pretrain_net_dir_for_test, pretrain_net_dir_for_align, DB_dir, noise_type, cuda_num=None):
    R_S_L_F_D_test.main2(pretrain_net_dir_for_test, pretrain_net_dir_for_align,
                         DB_dir, noise_type, cuda_num)

