# Real world Image Restoration

## Overview
rain, fog, dust, snow, lowlight(sRGB), lowlight(RAW-RGB)  

## Datasets
Dataset structure  
`DB_in_DataSet`  
  `├──01`   
  `├──02`   
  `├──...`   
  `└──08`   
      `├──D-210925_I2005L01_001.dng`   
      `├──D-210925_I2005L01_001_0001.json`   
      `└──...`   


## Installation
- Requirements

    ```
    colour
    colour_demosaicing
    numpy
    opencv_python
    Pillow
    PyYAML
    rawpy
    scikit_image
    skimage
    torch==1.7.1 (CUDA 11.0 is recommended)
    torchvision
    tqdm
    pyexiv2
    ```
  
- pyexiv2  
    It uses some C/C++ code and it needs other modules in C/C++. For this you have to install these.
    Install py3exiv2 not pyexiv2.
    ```
    apt install exiv2
    apt install python3-dev
    apt install libexiv2-dev
    apt install libboost-python-dev
    pip install py3exiv2
    ```   
  
## Installation inside specific environments - docker  

- doker image
   ```shell
   docker pull spnova12/img_restoration:latest
   ```
- example
   ```
  docker run -it --rm --gpus all --ipc=host -v /nvme2-2/human_and_forest/database/:/works/database -v /nvme2-2/human_and_forest/project:/works/project spnova12/img_restoration:latest
  cd /works/project/
  git clone https://github.com/spnova12/Real-world-Image-Restoration.git 
  cd Real-world-Image-Restoration/
  ```

  
## Quick Run

## Training and Evaluation
### 1. sRGB
1. Download Dataset.  

2. Set the '.dirs.yaml'.  
3. Init the dataset.  
    ```shell
    python init.py -mode sRGB
    ```  
4. Check dataset.  
    ```shell
    python get_data_info.py -mode sRGB
    ```
5. Train align network.
    ```shell
    python init.py -mode sRGB -train_align_net -exp_name {} -noise_type {} -cuda_num {}
    ```
   - noise type 
     - R : Rain  
     - F : Fog  
     - D : Dust   
     - S : Snow  
     - L : Lowlight    
   - cuda_num None -> It means use multi gpus.
   - example  
    ```shell
    # rain  
    python init.py -mode sRGB -train_align_net -exp_name rain001 -noise_type R -cuda_num 0  
   
    # fog  
    python init.py -mode sRGB -train_align_net -exp_name fog001 -noise_type F -cuda_num 1  
       
    # snow  
    python init.py -mode sRGB -train_align_net -exp_name snow001 -noise_type S -cuda_num 2  
   
    # lowlight  
    python init.py -mode sRGB -train_align_net -exp_name lowl001 -noise_type L -cuda_num 3  
   ```

### 2. rawRGB
1. Download Dataset.
2. Set the '.dirs.yaml'.
3. Init the dataset. 
    ```shell
    python init.py -mode rawRGB
    ```  
    Doing this you get 'DB_in_DataSet_patches'. 
    DNG images are split into patches(.DNG -> .bz2).   
4. After making the patch is done, check the refined Dataset result.
    ```shell
    python get_data_info.py -mode rawRGB
    ```
   Result bz2 patch samples are saved as 8bit sRGB through simple ISP
   in './rawRGB'.
5. Train.  
    If you have pretrained model then you can skip training.  
6. Test.  

## Results

## Contact 