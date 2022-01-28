rawRGB
---


### Overview

### Datasets

### Installation

### Quick Run
0. Preprocessing
```shell
# Split DNG images into patches(.DNG -> .bz2).
python init_raw_DB.py --DNGs_dir

# Load the patches and save the sample as an sRGB png.
# Sample name : input_patch_sample_sRGB_uint8.png, target_patch_sample_sRGB_uint8.png
python check_patches.py --DNGs_dir --Patches_dir

# Since the alignment of the dataset is not yet sufficient, 
# Run the code for align.
python check_patches_findbestpairs.py --DNGs_dir --Patches_dir
```

1. Train
```shell
python d0_raw.py
```

### Training and Evaluation

### Results

### Contact 



