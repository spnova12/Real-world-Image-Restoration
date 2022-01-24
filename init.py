import yaml

with open('dirs.yaml') as f:
    dirs = yaml.load(f, Loader=yaml.FullLoader)

print(dirs)