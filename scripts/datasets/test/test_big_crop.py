import os
import glob
import cv2

input_imgs_path = 'datasets/data_ovarian/test_input_rename_warped/' #test source dataset after rename. For B channel, 'datasets/data_ovarian/test_input_B_rename_warped/'
GT_imgs_path = 'datasets/data_ovarian/test_GT_rename/' #test target dataset after rename. For B channel, 'datasets/data_ovarian/test_GT_B_rename/'
new_input_imgs_path = 'datasets/data_ovarian/test_input_rename_warped_crop/'#For B channel, 'datasets/data_ovarian/test_input_B_rename_warped_crop/'
new_GT_imgs_path = 'datasets/data_ovarian/test_GT_rename_crop/'#For B channel, 'datasets/data_ovarian/test_GT_B_rename_crop/'
if not os.path.exists(new_input_imgs_path):
    os.makedirs(new_input_imgs_path)
if not os.path.exists(new_GT_imgs_path):
    os.makedirs(new_GT_imgs_path)

def crop_im(input_imgs_path,new_imgs_path,cover):
    im_files = sorted(list(glob.glob(os.path.join(input_imgs_path, '*.tif'))))
    for im_file0 in im_files:
        im_file = os.path.split(im_file0)[-1].split('.')[0]
        im = cv2.imread(im_file0, -1)
        h, w = im.shape[0:2]
        h_num, w_num = 4, 4
        hh, ww = h // h_num, w // w_num
        new_ims = h_num*w_num*[0]
        new_im_files = h_num*w_num*['']
        for i in range(h_num):
            for j in range(w_num):
                num = h_num*i+j
                if i==0:
                    h_start = hh * i
                elif i==h_num-1:
                    h_start = hh * i - 2*cover
                else:
                    h_start = hh * i - cover
                if j==0:
                    w_start = ww * j
                elif j==w_num-1:
                    w_start = ww * j - 2*cover
                else:
                    w_start = ww * j - cover
                if i==h_num-1:
                    h_end = hh * (i + 1)
                elif i==0:
                    h_end = hh * (i + 1) + 2*cover
                else:
                    h_end = hh * (i + 1) + cover
                if j==w_num-1:
                    w_end = ww * (j + 1)
                elif j==0:
                    w_end = ww * (j + 1) + 2*cover
                else:
                    w_end = ww * (j + 1) + cover
                new_ims[num] = im[h_start:h_end, w_start:w_end]
                new_im_files[num] = os.path.join(new_imgs_path, (im_file + '_' + str(num) + '.tif'))
                cv2.imwrite(new_im_files[num], new_ims[num])

#crop big image into 4*4 sub block, with pixel cover between each block (for deblock effect in inference)
cover = 32
crop_im(input_imgs_path,new_input_imgs_path,cover)
cover*=4
crop_im(GT_imgs_path,new_GT_imgs_path,cover)
print('done!')
