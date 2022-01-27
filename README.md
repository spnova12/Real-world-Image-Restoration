# Real world Image Restoration

  
rain, fog, dust, snow, lowlight(sRGB), lowlight(RAW-RGB)

## 1. sRGB
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



## 2. rawRGB
1. Download Dataset.  
Dataset structure  
`DB_in_DataSet` <br/>
  `├──01`  <br/>
  `├──02` <br/>
  `├──...` <br/>
  `└──08` <br/>
      `├──D-210925_I2005L01_001.dng` <br/>
      `├──D-210925_I2005L01_001_0001.json` <br/>
      `└──...`  


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
