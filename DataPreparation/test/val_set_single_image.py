import cv2
import glob
import random
import os


GT_img_root = 'datasets/data_ovarian/test_GT_rename/'#B channel: 'datasets/data_ovarian/test_GT_B/'
GT_val_img_root = 'datasets/data_ovarian/val_GT/'#B channel: 'datasets/data_ovarian/val_GT_B/'
if not os.path.exists(GT_val_img_root):
    os.makedirs(GT_val_img_root)

input_img_root = 'datasets/data_ovarian/test_input_rename_warped/'#B channel: 'datasets/data_ovarian/test_input_B/'
input_val_img_root = 'datasets/data_ovarian/val_input/'#B channel: 'datasets/data_ovarian/val_input_B/'
if not os.path.exists(input_val_img_root):
    os.makedirs(input_val_img_root)
img_paths = glob.glob(input_img_root+'*.tif')
for img_path in img_paths:
    img = cv2.imread(img_path)
    img_name = os.path.basename(img_path)
    img_GT = cv2.imread(os.path.join(GT_img_root,img_name))
    h,w = img.shape[0:2]
    img = img[h//2:-1,]
    # cv2.imshow('img', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    h, w = img_GT.shape[0:2]
    img_GT = img_GT[h//2:-1, ]

    h, w = img.shape[0:2]
    #random choose image block with source size 128*128, target size 512*512.
    for i in range(10):
        ran_h = random.randint(0,h-128)
        ran_w = random.randint(0,w-128)
        val_img = img[ran_h:ran_h+128,ran_w:ran_w+128]
        val_img_GT = img_GT[4*ran_h:4*ran_h+512,4*ran_w:4*ran_w+512]
        cv2.imwrite(os.path.join(input_val_img_root,(str(i+1) + '.tif')),val_img)
        cv2.imwrite(os.path.join(GT_val_img_root,(str(i+1) + '.tif')),val_img_GT)

