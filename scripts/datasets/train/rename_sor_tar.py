import  cv2
import glob
import os

train_test = 'train' #train or test set
dir_sou = 'datasets/data_ovarian/{}_input/'.format(train_test) #source data path. For B channel, 'datasets/data_ovarian/{}_input_B/'.format(train_test)
dir_tar = 'datasets/data_ovarian/{}_GT/'.format(train_test) #target data path. For B channel, 'datasets/data_ovarian/{}_GT_B/'.format(train_test)
new_dir_sou = 'datasets/data_ovarian/{}_input_rename'.format(train_test) #renamed source data path. For B channel, 'datasets/data_ovarian/{}_input_B_rename'.format(train_test)
new_dir_tar = 'datasets/data_ovarian/{}_GT_rename'.format(train_test) #renamed target data path. For B channel, 'datasets/data_ovarian/{}_GT_B_rename'.format(train_test)
if not os.path.exists(new_dir_sou):
    os.makedirs(new_dir_sou)
if not os.path.exists(new_dir_tar):
    os.makedirs(new_dir_tar)

imgs_s = glob.glob(os.path.join(dir_sou,'*tif'))
for img_s in imgs_s:
    img_name = os.path.basename(img_s)
    img_name = img_name.replace(' ', '').replace('1140nm', '')
    img_name, img_ext = os.path.splitext(img_name)
    img_name_ = img_name.split('resonant')
    img_name = img_name_[0]+img_name_[1][-2:]
    img2 = cv2.imread(img_s, 0)
    new_img_name = img_name + img_ext
    cv2.imwrite(os.path.join(new_dir_sou, new_img_name), img2)

imgs_t = glob.glob(os.path.join(dir_tar,'*tif'))
for img_t in imgs_t:
    img_name = os.path.basename(img_t)
    img_name = img_name.replace(' ','').replace('1140nm', '')
    img_name, img_ext = os.path.splitext(img_name)
    if 'galvano' in img_name:
        img_name_ = img_name.split('galvano')
    else:
        img_name_ = img_name.split('galvo')
    img_name = img_name_[0] + img_name_[1][-2:]
    img1 = cv2.imread(img_t, 0)
    new_img_name = img_name + img_ext
    cv2.imwrite(os.path.join(new_dir_tar, new_img_name), img1)