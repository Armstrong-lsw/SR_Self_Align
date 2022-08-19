import os
import glob
import cv2
from functools import cmp_to_key
import numpy as np


imgs_path = 'results/SR_Self_Align_x4_sesam/visualization/data_ovarian/'
result_imgs_path = 'results/stitched_imgaes/SR_Self_Align_x4_sesam/'

h_num, w_num = 24, 24
overlap = 64

overlap*=4  # 4 times pixel magnification


def my_compare(a, b):
    a_all = a.split('_')
    b_all = b.split('_')
    idx_a = a_all[-1]
    idx_b = b_all[-1]
    name_a = ''.join(a_all[:-1])
    name_b = ''.join(b_all[:-1])
    idx_a = int(idx_a)
    idx_b = int(idx_b)
    if name_a > name_b:
        return 1
    elif name_a < name_b:
        return -1
    else:
        if idx_a>idx_b:
            return 1
        else:
            return -1



if not os.path.exists(result_imgs_path):
    os.makedirs(result_imgs_path)
im_files = sorted(list(glob.glob(os.path.join(imgs_path, '*.png'))))
# badcase = ['BorderlineA1-2', 'IAA2-8', 'ICA1-6', 'IVA1-4']

suffix = '_SR_Self_Align_x4_sesam' #For B channel, '_SR_Self_Align_x4_B_sesam'
#sort result sub image in turns of _0,_1,_2,...,_15
im_names = [os.path.split(im_file0)[-1].split('.')[0].replace(suffix, '') for im_file0 in im_files]
im_names_files = {}
im_names_ = []
for im_name, im_file in zip(im_names, im_files):
    im_names_files[im_name] = im_file
    im_names_.append(im_name)
im_names = im_names_
im_names = sorted(im_names, key=cmp_to_key(my_compare))

h_s,w_s = -1, -1
h_t,w_t = -1, -1
#stitch sub image into origin size
for i_, im_name in enumerate(im_names):
    im_file0 = im_names_files[im_name]
    im = cv2.imread(im_file0)
    im_num = int(im_name.split('_')[-1])
    i = im_num//w_num   #i row idx
    j = im_num%w_num    #j col idx
    if overlap != 0:
        if i == 0:
            im = im[:-2 * overlap, :]
        elif i==h_num-1:
            im = im[2 * overlap:, :]
        else:
            im = im[overlap:-overlap, :]
        if j == 0:
            im = im[:, :-2 * overlap]
        elif j==w_num-1:
            im = im[:, 2 * overlap:]
        else:
            im = im[:, overlap:-overlap]
    h, w = im.shape[0:2]
    if i_ % (h_num * w_num) == 0:
        h_s, w_s = h, w
        h_t, w_t = h_num * h_s, w_num * w_s
        im_t = np.zeros(shape=(h_t, w_t, 3))
    assert h == h_s, w == w_s
    im_t[h_s*i:h_s*(i+1),w_s*j:w_s*(j+1)] = im
    if (i_+1)%(h_num*w_num)==0:
        im_name_t = '_'.join(im_name.split('_')[:-1])
        im_name_t_path = os.path.join(result_imgs_path, im_name_t+'.png')
        cv2.imwrite(im_name_t_path, im_t)
        print('saving im %d'%(i_//(h_num*w_num)))

print('done!')
