import argparse

import sRGB.R_S_L_F_D.R_S_L_F_D_test as R_S_L_F_D_test


def test_my_db(pretrain_net_dir_for_test, test_DB_dir_list, out_dir_name, cuda_num):
    R_S_L_F_D_test.main(pretrain_net_dir_for_test, test_DB_dir_list, out_dir_name, cuda_num)




