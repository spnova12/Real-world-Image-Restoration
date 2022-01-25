from rawRGB.preprocessing.init_raw_DB import init_raw_DB
from rawRGB.preprocessing.check_patches_findbestpairs import check_patches_findbestpairs

def preprocessing(DB_dir):
    # (1) find json error, find DB errors, delete bad GTs, generate patches.
    # init_raw_DB(DB_dir)

    # (2) delete bad noises.
    check_patches_findbestpairs(DB_dir)
