##DataPreparation: <br> 
we merge 3 channels images into RGB image for display, c1 for B channel, c2 for G channel, c3 for R channel.<br>

For training, we merge R and G channel images and train one SR model; train another model for B channel images.
Because B channel images data preparation is diffrent from R and G in orb_match step. <br>

 Please right click the python files in the "DataPreparation/datasets/train and /test" to modify run configuration to change the working direction to root path).

### For training

For data preparation, we follow these steps.
1. For convenience, we rename raw dataset image name to ensure that the input and target sets have the same paired names [can use other approaches to do this]:
    ```
    DataPreparation/datasets/train/1_rename_sor_tar.py
    ```
   
2. Image warping with ORB (preregistration):
    ```
    DataPreparation/datasets/train/2_orb_warping.py
    ```

3. Crop edge pixel after warping due to preregistration:
    ```
    DataPreparation/datasets/train/3_crop_sor_tar.py
    ```


4. The training patches are usually small (In our training, LR: 64x64, HR: 256x256 for low-high resolution training patch). So there is a waste if reading the whole image but only using a very small part of it. In order to accelerate the IO speed during training, we crop the 2K->8K resolution image pairs to sub-image pairs 
Note that the size of sub-images is different from the training patch size (`gt_size`) defined in the config file. Specifically, the cropped sub-images with 512x512 are stored. The dataloader will further randomly crop the sub-images to `GT_size x GT_size` patches for training. <br/>
    
    ```
    DataPreparation/datasets/train/4_split_subimages_sor_tar.py
    ```

5. [Optional] Create LMDB files (if you dont want to use lmdb, simply change "type: lmdb" to "type: disk" in the train yml file). 
   ```
   DataPreparation/datasets/train/python scripts/5_create_lmdb_sor_tar.py
   ```

6. Make validation set after [For testing] Step. 3.


### For testing
1. For convenience, we rename raw dataset image name to ensure that the input and target sets have the same paired names [can use other approaches to do this]:
   
2. [Optional] Image warping with ORB (preregistration)::
    ```
    DataPreparation/datasets/test/2_test_orb_warping.py
    ```

3. Make validation set. Random crop test set blocks as val set.
    ```
    DataPreparation/datasets/test/3_val_set.py
    ```

4. [Optional] Split big images into image tiles for memory limit: 
    ```
   DataPreparation/datasets/test/4_test_split_images.py
   ```
   
5. [Optional] After inference, stitch image tiles into origin size: 
    ```
    DataPreparation/datasets/test/5_test_stitch_tiles.py
    ```
