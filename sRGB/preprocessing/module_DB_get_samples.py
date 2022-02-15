import tqdm
from sRGB.preprocessing.module_DB_manager import *

def get_DB_high_level_samples(total_dict, img_name_to_find=None):
    ############################################################################################################
    # Save representative sample.
    ############################################################################################################
    if img_name_to_find is None:
        print('\n:: Save representative(highest level) 3 samples')
    else:
        print('\n:: Save representative(highest level) All images with name')


    for my_key in tqdm.tqdm(total_dict.keys()):
        for my_key2 in total_dict[my_key].keys():
            if total_dict[my_key][my_key2]:

                levels = list(total_dict[my_key][my_key2].keys())
                levels.remove('GT')

                levels = [level for level in levels if len(total_dict[my_key][my_key2][level]) > 10]
                if levels:
                    h_level = sorted(levels)[-1]

                    noisy_samples = total_dict[my_key][my_key2][h_level]

                    if img_name_to_find is None:
                        # copy samples
                        db_samples = make_dirs(f'test-out/DB_high_level_samples/{my_key2}')
                        for n_sample in noisy_samples[:3]:
                            n_sample_bname = os.path.basename(n_sample)
                            copyfile(f'{n_sample}', f'{db_samples}/{n_sample_bname}')
                    else:
                        # example : D-211010_O1024S04_001_0001
                        if img_name_to_find in noisy_samples[0]:
                            # copy samples
                            db_samples = make_dirs(f'test-out/DB_high_level_samples_{img_name_to_find}')
                            for n_sample in noisy_samples:
                                n_sample_bname = os.path.basename(n_sample)
                                copyfile(f'{n_sample}', f'{db_samples}/{n_sample_bname}')