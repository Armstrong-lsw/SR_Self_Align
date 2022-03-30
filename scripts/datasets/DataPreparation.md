##DataPreparation: <br> 
we merge 3 channels images into RGB image for display, c1 for B channel, c2 for G channel, c3 for R channel.<br>

For training, we merge R and G channel images and train one SR model; train another model for B channel images.
Because R channel images data preparation is diffrent from B and G in orb_match step. <br>

### For training

for data preparation, we follow these steps.
1. For convenience, we rename raw dataset image name [can use other approaches to do this]:
    ```
    scripts/datasets/train/rename_sor_tar.py
    ```
   
2. Image warping with orb:
    ```
    scripts/datasets/train/orb_warping.py
    ```

3. Crop edge pixel after warping:
    ```
    scripts/datasets/train/crop_sor_tar.py
    ```


4. The training patches are usually small (In our training, LR: 64x64, HR: 256x256 for low-high resolution training patch). So there is a waste if reading the whole image but only using a very small part of it. In order to accelerate the IO speed during training, we crop the 2K->8K resolution image pairs to sub-image pairs 
Note that the size of sub-images is different from the training patch size (`gt_size`) defined in the config file. Specifically, the cropped sub-images with 512x512 are stored. The dataloader will further randomly crop the sub-images to `GT_size x GT_size` patches for training. <br/>
    
    ```
    scripts/datasets/train/split_subimages_sor_tar.py
    ```

5. [Optional] Create LMDB files. Please refer to [LMDB Description](#LMDB-Description). 
   ```
   scripts/datasets/train/python scripts/create_lmdb_sor_tar.py
   ```

6. make validation set after [### For testing] Step. 4.


### For testing
1. For convenience, we rename raw dataset image name [can use other approaches to do this]:
   
2. Image warping with orb:
    ```
    scripts/datasets/test/test_orb_warping.py
    ```

3. Make validation set. Random crop test set blocks as val set.
    ```
    scripts/datasets/test/val_set.py
    ```

4. Split big images into image tiles for memory limit: 
    ```
   scripts/datasets/test/test_split_images.py
   ```
   
5. After inference, stitch image tiles into origin size: 
    ```
    scripts/datasets/test/test_stitch_tiles.py
    ```
