# Training and Testing

[English](TrainTest.md) **|** [简体中文](TrainTest_CN.md)

Please run the commands in the root path of `BasicSR`. <br>
In general, both the training and testing include the following steps:

1. Prepare datasets. Please refer to [DataPreparation_For_Ovarian](scripts/datasets/DataPreparation_For_Ovarian.md) and [DatasetPreparation.md](DatasetPreparation.md)
1. Modify config files. The config files are under the `options` folder. For more specific configuration information, please refer to [Config](Config.md)
1. [Optional] You may need to download pre-trained models if you are testing or using pre-trained models. Please see [ModelZoo](ModelZoo.md)
1. Run commands. Use [Training Commands](#Training-Commands) or [Testing Commands](#Testing-Commands) accordingly.

#### 目录

1. [Training Commands](#Training-Commands)
    1. [Single GPU Training](#Single-GPU-Training)
    1. [Distributed (Multi-GPUs) Training](#Distributed-Training)
    1. [Slurm Training](#Slurm-Training)
1. [Testing Commands](#Testing-Commands)
    1. [Single GPU Testing](#Single-GPU-Testing)
    1. [Distributed (Multi-GPUs) Testing](#Distributed-Testing)
    1. [Slurm Testing](#Slurm-Testing)

## Training Commands

### Single GPU Training

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0 \\\
> python basicsr/train.py -opt options/train/RRDB/train_SR_Self_Align_sesam.yml

### Distributed Training

**8 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\\
> python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/RRDB/train_SR_Self_Align_sesam.yml --launcher pytorch

**4 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0,1,2,3 \\\
> python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/RRDB/train_SR_Self_Align_sesam.yml --launcher pytorch

## Testing Commands

### Single GPU Testing

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0 \\\
> python basicsr/test.py -opt options/test/RRDB/test_SR_Self_Align_sesam.yml

### Distributed Testing

**8 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\\
> python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/test.py -opt options/test/RRDB/test_SR_Self_Align_sesam.yml --launcher pytorch

**4 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0,1,2,3 \\\
> python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/test.py -opt options/test/RRDB/test_SR_Self_Align_sesam.yml  --launcher pytorch
