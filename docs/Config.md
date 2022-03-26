# Configuration

[English](Config.md) **|** [简体中文](Config_CN.md)

#### Contents

[Configuration Explanation](#Configuration-Explanation)<br>
1. [Training Configuration](#Training-Configuration)
1. [Testing Configuration](#Testing-Configuration)


**Note**: If `debug` is in the experiment name, it will enter the debug mode. That is, the program will log and validate more intensively and will not use `tensorboard logger` and `wandb logger`.

## Configuration Explanation

We use yaml files for configuration.

### Training Configuration

Taking [SR_Self_Align_x4_sesam.yml](../options/train/RRDB/train_SR_Self_Align_sesam.yml) as an example:

```yml
####################################
# The following are general settings
####################################
# Model type. Usually the class name defined in the `models` folder
model_type: SRGANModelDeform
# The upsampling ratio. If not defined, use 1
scale: 4
# The number of GPUs for training
num_gpu: 1  # set num_gpu: 0 for cpu mode
# Random seed
manual_seed: 0

########################################################
# The following are the dataset and data loader settings
########################################################
datasets:
  # Training dataset settings
  train:
    # Dataset name
    name: data_ovarian
    # Dataset type. Usually the class name defined in the `data` folder
    type: PairedImageDataset
    #### The following arguments are flexible and can be obtained in the corresponding doc
    # GT (Ground-Truth) folder path
    dataroot_gt: datasets/data_ovarian/test_GT_rename_crop/
    # LQ (Low-Quality) folder path
    dataroot_lq: datasets/data_ovarian/test_input_rename_warped_crop/
    # template for file name. Usually, LQ files have suffix like `_x4`. It is used for file name mismatching
    filename_tmpl: '{}'
    # IO backend, more details are in [docs/DatasetPreparation.md]
    io_backend:
      # directly read from disk
      type: disk

    # Ground-Truth training patch size
    gt_size: 256
    # Whether to use horizontal flip. Here, flip is for horizontal flip
    use_flip: true
    # Whether to rotate. Here for rotations with every 90 degree
    use_rot: true

    #### The following are data loader settings
    # Whether to shuffle
    use_shuffle: true
    # Number of workers of reading data for each GPU
    num_worker_per_gpu: 6
    # Total training batch size
    batch_size_per_gpu: 8
    # THe ratio of enlarging dataset. For example, it will repeat 100 times for a dataset with 15 images
    # So that after one epoch, it will read 1500 times. It is used for accelerating data loader
    # since it costs too much time at the start of a new epoch
    dataset_enlarge_ratio: 100

  # validation dataset settings
  val:
    # Dataset name
    name: data_ovarian
    # Dataset type. Usually the class name defined in the `data` folder
    type: PairedImageDataset
    #### The following arguments are flexible and can be obtained in the corresponding doc
    # GT (Ground-Truth) folder path
    dataroot_gt: datasets/data_ovarian/val_GT
    # LQ (Low-Quality) folder path
    dataroot_lq: datasets/data_ovarian/val_input
    # IO backend, more details are in [docs/DatasetPreparation.md]
    io_backend:
      # directly read from disk
      type: disk

##################################################
# The following are the network structure settings
##################################################
# network g settings
network_g:
  # Architecture type. Usually the class name defined in the `models/archs` folder
  type: RRDBDCNNetSESAM
  #### The following arguments are flexible and can be obtained in the corresponding doc
  # Channel number of inputs
  num_in_ch: 1
  # Channel number of outputs
  num_out_ch: 1
  # Channel number of middle features
  num_feat: 64
  # block number
  num_block: 20
  # Channel number of RRDB grow features
  num_grow_ch: 32
  # Deformable groups, defaults:8
  deformable_group: 8
  # Residual blocks number for feature extraction
  num_extract_block: 10

# network d settings
network_d:
  # Architecture type. Usually the class name defined in the `models/archs` folder
  type: Discriminator
  #### The following arguments are flexible and can be obtained in the corresponding doc
  # Channel number of inputs
  in_channels: 1
  # Whether to use sigmoid after last convolution in discriminator. so False.
  use_sigmoid: False

#########################################################
# The following are path, pretraining and resume settings
#########################################################
path:
  # Path for pretrained models, usually end with pth
  pretrain_network_g: experiments_pretrain/SR_Self_Align_x4_sesam/models/net_g_110000.pth
  # Whether to load pretrained models strictly, that is the corresponding parameter names should be the same
  strict_load_g: true
  # Path for resume state. Usually in the `experiments/exp_name/training_states` folder
  # This argument will over-write the `pretrain_network_g`
  resume_state: ~


#####################################
# The following are training settings
#####################################
train:
  # Optimizer settings for generator
  optim_g:
    # Optimizer type
    type: Adam
    #### The following arguments are flexible and can be obtained in the corresponding doc
    # Learning rate
    lr: !!float 2e-4
    weight_decay: 0
    # beta1 and beta2 for the Adam
    betas: [0.9, 0.99]
  
  # Optimizer setting for discriminator
  optim_d: 
  # Learning rate scheduler settings
  scheduler:
    # Scheduler type
    type: CosineAnnealingRestartLR
    #### The following arguments are flexible and can be obtained in the corresponding doc
    # Cosine Annealing periods
    periods: [250000, 250000, 250000, 250000]
    # Cosine Annealing restart weights
    restart_weights: [1, 1, 1, 1]
    # Cosine Annealing minimum learning rate
    eta_min: !!float 1e-7

  # Total iterations for training
  total_iter: 400000
  # Warm up iterations. -1 indicates no warm up
  warmup_iter: -1

  #### The following are loss settings
  # Pixel-wise loss options
  pixel_opt:
    # Loss type. Usually the class name defined in the `basicsr/models/losses` folder
    type: L1Loss
    # Loss weight
    loss_weight: 1.0
    # Loss reduction mode
    reduction: mean
  #perceptual loss options 
  perceptual_opt:
    # Loss type. The class name defined in the `basicsr/models/losses/losses.py` 
    type: PerceptualLoss
    # Perceptual layer and weight.
    layer_weights:
      'conv5_4': 1  # before relu
    # Perceptual feature extractor network.
    vgg_type: vgg19
    # Whether to normalize the perceptual network input image. Default:True
    use_input_norm: true
    #Perceptual loss weight.
    perceptual_weight: 0.1
    #Style loss weight. Default:0, that is no style loss.
    style_weight: 0
    #Whether to norm the perceptual network input image to [0,1], this is different from `use_input_norm`.  
    norm_img: false
    # Criterion used for perceptual loss. Default: 'l1'.
    criterion: l1
  # GAN loss options 
  gan_opt:
    #GAN loss type. The class name defined in the `basicsr/models/losses/losses.py`
    type: GANLoss
    gan_type: vanilla
    #The value for real label. Default: 1.0. 
    real_label_val: 1.0
    #The value for fake label. Default: 0.0. 
    fake_label_val: 0.0
    #Gan loss weight in generator. 
    loss_weight: !!float 5e-2

#######################################
# The following are validation settings
#######################################
val:
  # validation frequency. Validate every 5000 iterations
  val_freq: !!float 5e3
  # Whether to save images during validation
  save_img: false

  # Metrics in validation
  metrics:
    # Metric name. It can be arbitrary
    psnr:
      # Metric type. Usually the function name defined in the`basicsr/metrics` folder
      type: calculate_psnr
      #### The following arguments are flexible and can be obtained in the corresponding doc
      # Whether to crop border during validation
      crop_border: 4
      # Whether to convert to Y(CbCr) for validation
      test_y_channel: false

########################################
# The following are the logging settings
########################################
logger:
  # Logger frequency
  print_freq: 100
  # The frequency for saving checkpoints
  save_checkpoint_freq: !!float 5e3
  # Whether to tensorboard logger
  use_tb_logger: true
  # Whether to use wandb logger. Currently, wandb only sync the tensorboard log. So we should also turn on tensorboard when using wandb
  wandb:
    # wandb project name. Default is None, that is not using wandb.
    # Here, we use the basicsr wandb project: https://app.wandb.ai/xintao/basicsr
    project: basicsr
    # If resuming, wandb id could automatically link previous logs
    resume_id: ~

################################################
# The following are distributed training setting
# Only require for slurm training
################################################
dist_params:
  backend: nccl
  port: 29500
```

### Testing Configuration

Taking [test_SR_Self_Align_sesam.yml](../options/test/RRDB/test_SR_Self_Align_sesam.yml) as an example:

```yml
# Experiment name
name: SR_Self_Align_x4_sesam
# Model type. Usually the class name defined in the `models` folder
model_type: SRGANModelDeform
# The upsampling ratio. If not defined, use 1
scale: 4
# The number of GPUs for testing
num_gpu: 1  # set num_gpu: 0 for cpu mode

########################################################
# The following are the dataset and data loader settings
########################################################
datasets:
  # Testing dataset settings. The first testing dataset
  test_1:
    # Dataset name
    name: data_ovarian
    # Dataset type. Usually the class name defined in the `data` folder
    type: PairedImageDataset
    #### The following arguments are flexible and can be obtained in the corresponding doc
    # GT (Ground-Truth) folder path
    dataroot_gt: datasets/data_ovarian/test_GT_rename_crop/
    # LQ (Low-Quality) folder path
    dataroot_lq: datasets/data_ovarian/test_input_rename_warped_crop/
    # IO backend, more details are in [docs/DatasetPreparation.md]
    io_backend:
      # directly read from disk
      type: disk
  # Testing dataset settings. The second testing dataset

##################################################
# The following are the network structure settings
##################################################
# network g settings
network_g:
  # Architecture type. Usually the class name defined in the `models/archs` folder
  type: RRDBDCNNetSESAM
  #### The following arguments are flexible and can be obtained in the corresponding doc
  # Channel number of inputs
  num_in_ch: 1
  # Channel number of outputs
  num_out_ch: 1
  # Channel number of middle features
  num_feat: 64
  # block number
  num_block: 20
  # Channel number of RRDB grow features
  num_grow_ch: 32
  # Deformable groups, defaults:8
  deformable_group: 8
  # Residual blocks number for feature extraction
  num_extract_block: 10

#################################################
# The following are path and pretraining settings
#################################################
path:
  ## Path for pretrained models, usually end with pth. Our pretrain model link (https://drive.google.com/file/d/1-0UHFuJ9qR5tTQqLq0KvQ5CCzvY3q8il/view?usp=sharing)
  pretrain_network_g: ./experiments_pretrain/SR_Self_Align_x4_sesam_net_g.pth
  # Whether to load pretrained models strictly, that is the corresponding parameter names should be the same
  strict_load_g: true

##########################################################
# The following are validation settings (Also for testing)
##########################################################
val:
  # Whether to save images during validation
  save_img: true
  # Suffix for saved images. If None, use exp name
  suffix: ~

  # Metrics in validation
  metrics:
    # Metric name. It can be arbitrary
    psnr:
      # Metric type. Usually the function name defined in the`basicsr/metrics` folder
      type: calculate_psnr
      #### The following arguments are flexible and can be obtained in the corresponding doc
      # Whether to crop border during validation
      crop_border: 4
      # Whether to convert to Y(CbCr) for validation
      test_y_channel: false
    # Another metric
    ssim:
      type: calculate_ssim
      crop_border: 4
      test_y_channel: false
```
