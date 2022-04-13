import os
import glob
import cv2

input_imgs_path = 'datasets/data_ovarian/test_input_rename_warped'
input_tiles_path = 'datasets/data_ovarian/test_input_rename_warped_tiles'
GT_imgs_path = 'datasets/data_ovarian/test_GT_rename'
GT_tiles_path = 'datasets/data_ovarian/test_GT_rename_tiles'

h_num, w_num = 4, 4
overlap = 0

if not os.path.exists(input_tiles_path):
    os.makedirs(input_tiles_path)
if not os.path.exists(GT_tiles_path):
    os.makedirs(GT_tiles_path)


def crop_im(imgs_path, tiles_path, overlap):
    im_files = sorted(list(glob.glob(os.path.join(imgs_path, '*.tif'))))
    for im_file0 in im_files:
        im_file = os.path.split(im_file0)[-1].split('.')[0]
        im = cv2.imread(im_file0, 1)
        h, w = im.shape[0:2]

        hh, ww = h // h_num, w // w_num
        new_ims = h_num * w_num * [0]
        new_im_files = h_num * w_num * ['']
        for i in range(h_num):
            for j in range(w_num):
                num = h_num * i + j
                if i == 0:
                    h_start = hh * i
                elif i == h_num - 1:
                    h_start = hh * i - 2 * overlap
                else:
                    h_start = hh * i - overlap
                if j == 0:
                    w_start = ww * j
                elif j == w_num - 1:
                    w_start = ww * j - 2 * overlap
                else:
                    w_start = ww * j - overlap
                if i == h_num - 1:
                    h_end = hh * (i + 1)
                elif i == 0:
                    h_end = hh * (i + 1) + 2 * overlap
                else:
                    h_end = hh * (i + 1) + overlap
                if j == w_num - 1:
                    w_end = ww * (j + 1)
                elif j == 0:
                    w_end = ww * (j + 1) + 2 * overlap
                else:
                    w_end = ww * (j + 1) + overlap
                new_ims[num] = im[h_start:h_end, w_start:w_end]
                new_im_files[num] = os.path.join(tiles_path, (im_file + '_' + str(num) + '.tif'))
                cv2.imwrite(new_im_files[num], new_ims[num])


crop_im(input_imgs_path, input_tiles_path, overlap)
overlap *= 4
crop_im(GT_imgs_path, GT_tiles_path, overlap)
print('done!')
