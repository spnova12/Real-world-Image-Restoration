import yaml

from rawRGB.preprocessing_ import preprocessing

with open('dirs.yaml') as f:
    dirs = yaml.load(f, Loader=yaml.FullLoader)



preprocessing(dirs['rawRGB']['DB_dir'])


