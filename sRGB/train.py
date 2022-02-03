
import sRGB.R_S_L_F_D.R_S_L_F_D_train as R_S_L_F_D_train


def train(exp_name, DB_dir, noise_type, pretrain_net_dir_for_align, cuda_num=None):
    R_S_L_F_D_train.main(exp_name, DB_dir, noise_type, pretrain_net_dir_for_align, cuda_num)