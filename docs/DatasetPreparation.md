# Dataset Preparation

[English](DatasetPreparation.md)

#### Contents

1. [Data Storage Format](#Data-Storage-Format)
    1. [How to Use](#How-to-Use)
    1. [How to Implement](#How-to-Implement)
    1. [LMDB Description](#LMDB-Description)
    1. [Data Pre-fetcher](#Data-Pre-fetcher)
1. [Image Super-Resolution](#Image-Super-Resolution)
    1. [data_ovarian](#data_ovarian)
    1. [Common Image SR Datasets](#Common-Image-SR-Datasets)

## Data Storage Format

At present, there are two types of data storage formats supported:

1. Store in `hard disk` directly in the format of images.
1. Make [LMDB](https://lmdb.readthedocs.io/en/release/), which could accelerate the IO and decompression speed during training.

#### How to Use

At present, we can modify the configuration yaml file to support different data storage formats. Taking [PairedImageDataset](../basicsr/data/paired_image_dataset.py) as an example, we can modify the yaml file according to different requirements.

1. Directly read disk data.

    ```yaml
    type: PairedImageDataset
    dataroot_gt: datasets/data_ovarian/train_GT_rename_crop_sub
    dataroot_lq: datasets/data_ovarian/train_input_rename_warped_crop_sub
    io_backend:
      type: disk
    ```

1. Use LMDB.
We need to make LMDB before using it. Please refer to [LMDB description](#LMDB-Description). Note that we add meta information to the original LMDB, and the specific binary contents are also different. Therefore, LMDB from other sources can not be used directly.

    ```yaml
    type: PairedImageDataset
    dataroot_gt: datasets/data_ovarian/train_GT_rename_crop_sub.lmdb
    dataroot_lq: datasets/data_ovarian/train_input_rename_warped_crop_sub.lmdb
    io_backend:
      type: lmdb
    ```
   
#### How to Implement

The implementation is to call the elegant fileclient design in [mmcv](https://github.com/open-mmlab/mmcv). In order to be compatible with BasicSR, we have made some changes to the interface (mainly to adapt to LMDB). See [file_client.py](../basicsr/utils/file_client.py) for details.

When we implement our own dataloader, we can easily call the interfaces to support different data storage forms. Please refer to [PairedImageDataset](../basicsr/data/paired_image_dataset.py) for more details.

#### LMDB Description

During training, we use LMDB to speed up the IO and CPU decompression. (During testing, usually the data is limited and it is generally not necessary to use LMDB). The acceleration depends on the configurations of the machine, and the following factors will affect the speed:

1. Some machines will clean cache regularly, and LMDB depends on the cache mechanism. Therefore, if the data fails to be cached, you need to check it. After the command `free -h`, the cache occupied by LMDB will be recorded under the `buff/cache` entry.
1. Whether the memory of the machine is large enough to put the whole LMDB data in. If not, it will affect the speed due to the need to constantly update the cache.
1. If you cache the LMDB dataset for the first time, it may affect the training speed. So before training, you can enter the LMDB dataset directory and cache the data by: ` cat data.mdb > /dev/nul`.

In addition to the standard LMDB file (data.mdb and lock.mdb), we also add `meta_info.txt` to record additional information.
Here is an example:

**Folder Structure**

```txt
train_GT_rename_crop_sub.lmdb
├── data.mdb
├── lock.mdb
├── meta_info.txt
```

**meta information**

`meta_info.txt`, We use txt file to record for readability. The contents are:

```txt
Ovarianborderlinea1-5c2_s001.png (512,512,1) 1
Ovarianborderlinea1-5c2_s002.png (512,512,1) 1
Ovarianborderlinea1-5c2_s003.png (512,512,1) 1
Ovarianborderlinea1-5c2_s004.png (512,512,1) 1
...
```

Each line records an image with three fields, which indicate:

- Image name (with suffix): Ovarianborderlinea1-5c2_s001.png
- Image size: (512, 512,1) represents a 512x512x1 image
- Other parameters (BasicSR uses cv2 compression level for PNG): In restoration tasks, we usually use PNG format, so `1` represents the PNG compression level `CV_IMWRITE_PNG_COMPRESSION` is 1. It can be an integer in [0, 9]. A larger value indicates stronger compression, that is, smaller storage space and longer compression time.

**Binary Content**

For convenience, the binary content stored in LMDB dataset is encoded image by cv2: `cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level]`. You can control the compression level by `compress_level`, balancing storage space and the speed of reading (including decompression).

**How to Make LMDB**
We provide a script to make LMDB. Before running the script, we need to modify the corresponding parameters accordingly, that is R, G channel or B channel. <br>
 `python scripts/create_lmdb_sor_tar.py`

## Image Super-Resolution

It is recommended to symlink the dataset root to `SR_Self_Align/datasets` with the command `ln -s xxx yyy`. If your folder structure is different, you may need to change the corresponding paths in config files.

### data_ovarian

[ovarian_SR_dataset](https://drive.google.com/file/d/1-0AukQ7ffH-njMtt70unDxUcchfqUphx/view?usp=sharing) is a dataset of pairs of resonant scanning images(low-resolution) and galvanometer scanning images(high-resolution) for real image super-resolution task. 

**Preparation Steps**

1. Crop to sub-images: After orb_match and edge-crop, data_ovarian has 2K->8K low-high resolution (e.g., LR: 2176 × 2176, HR: 8704 x 8704) image pais. but the training patches are usually small (In our training, LR: 64x64, HR: 256x256 for low-high resolution training patch). So there is a waste if reading the whole image but only using a very small part of it. In order to accelerate the IO speed during training, we crop the 2K->8K resolution image pairs to sub-image pairs (here, we crop to 128x128->512x512 sub-image pairs with overlap ), then crop to 64x64 and corresponding 256x256 training patchs from 128x128->512x512 sub-image pairs by dataloader. R, G channel data for one training set, B channel for another. <br>
Note that the size of sub-images is different from the training patch size (`gt_size`) defined in the config file. Specifically, the cropped sub-images with 512x512 are stored. The dataloader will further randomly crop the sub-images to `GT_size x GT_size` patches for training. <br/>
    Run the script [extract_subimages_sor_tar.py](../scripts/extract_subimages_sor_tar.py):

    ```python
    python scripts/extract_subimages_sor_tar.py
    ```

    Remember to modify the paths and configurations if you have different settings.
1. [Optional] Create LMDB files. Please refer to [LMDB Description](#LMDB-Description). `python scripts/create_lmdb_sor_tar.py`. Use the `create_lmdb_for_ovarian` function and remember to modify the paths and configurations accordingly.
