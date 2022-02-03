
import sRGB.R_S_L_F_D.R_S_L_F_D_train as R_S_L_F_D_train


def train(exp_name, DB_dir, noise_type, pre_trained_align_net_pth, cuda_num=None):
    R_S_L_F_D_train.main(exp_name, DB_dir, noise_type, pre_trained_align_net_pth, cuda_num)