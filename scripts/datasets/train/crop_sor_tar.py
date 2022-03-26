import  cv2
import glob
import os

train_test='train' # test for testset
dir_sou = 'datasets/data_ovarian/{}_input_rename_warped'.format(train_test)#B channel: 'datasets/data_ovarian/{}_input_rename_warped'.format(train_test)
dir_tar = 'datasets/data_ovarian/{}_GT_rename'.format(train_test)#B channel: 'datasets/data_ovarian/{}_GT_B_rename'.format(train_test)
new_dir_sou = 'datasets/data_ovarian/{}_input_rename_warped_crop'.format(train_test)#B channel: 'datasets/data_ovarian/{}_input_B_rename_warped_crop'.format(train_test)
new_dir_tar = 'datasets/data_ovarian/{}_GT_rename_crop'.format(train_test)#B channel: 'datasets/data_ovarian/{}_GT_B_rename_crop'.format(train_test)
if not os.path.exists(new_dir_sou):
    os.makedirs(new_dir_sou)
if not os.path.exists(new_dir_tar):
    os.makedirs(new_dir_tar)
crop_width = 16
crop_width_t = 16*4
imgs_s = glob.glob(os.path.join(dir_sou,'*tif'))
for img_s in imgs_s:
    img_name = os.path.basename(img_s)
    img_t = os.path.join(dir_tar, img_name)
    img1 = cv2.imread(img_t, 0)
    img2 = cv2.imread(img_s, 0)
    img1_croped = img1[crop_width_t:-crop_width_t, crop_width_t:-crop_width_t]
    img2_croped = img2[crop_width:-crop_width, crop_width:-crop_width]
    cv2.imwrite(os.path.join(new_dir_sou, img_name), img2_croped)
    cv2.imwrite(os.path.join(new_dir_tar, img_name), img1_croped)
