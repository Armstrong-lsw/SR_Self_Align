import cv2
import glob
import random
import os

#choose val set from test set. for RG channel and B channel.
GT_img_root = 'datasets/data_ovarian/test_GT/'#B channel: 'datasets/data_ovarian/test_GT_B/'
GT_val_img_root = 'datasets/data_ovarian/val_GT/'#B channel: 'datasets/data_ovarian/val_GT_B/'
if not os.path.exists(GT_val_img_root):
    os.makedirs(GT_val_img_root)

input_img_root = 'datasets/data_ovarian/test_input/'#B channel: 'datasets/data_ovarian/test_input_B/'
input_val_img_root = 'datasets/data_ovarian/val_input/'#B channel: 'datasets/data_ovarian/val_input_B/'
if not os.path.exists(input_val_img_root):
    os.makedirs(input_val_img_root)
img_paths = glob.glob(input_img_root+'*.tif')
for img_path in img_paths:
    img = cv2.imread(img_path)
    img_name = os.path.basename(img_path)
    img_GT = cv2.imread(os.path.join(GT_img_root,img_name))
    h,w = img.shape[0:2]
    #random choose image block with source size 128*128, target size 512*512.
    ran_h = random.randint(0,h-128)
    ran_w = random.randint(0,w-128)
    val_img = img[ran_h:ran_h+128,ran_w:ran_w+128]
    val_img_GT = img_GT[4*ran_h:4*ran_h+512,4*ran_w:4*ran_w+512]
    cv2.imwrite(os.path.join(input_val_img_root,img_name),val_img)
    cv2.imwrite(os.path.join(GT_val_img_root,img_name),val_img_GT)