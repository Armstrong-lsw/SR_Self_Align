import numpy as np
import cv2
from grade_mode import grade_mode
import glob
import os

def drawMatches(img1, kp1, img2, kp2, matches, train_test, img_name):
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    warpImg_rgb = np.zeros([rows2, cols2])
    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0)
    warpImg = cv2.warpPerspective(img2, np.array(M), (img2.shape[1], img2.shape[0]),
                                  flags=cv2.WARP_INVERSE_MAP)
    img_path = 'datasets/data_ovarian/{}_input_B_rename_warped/'.format(train_test) #save warped source B channel data path.
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    cv2.imwrite(img_path+img_name, warpImg)

#process c1(B) channel, select c2(G) channel as assist channel, because B channel is too noisy to orb-match
train_test = 'train' #or test
channel_assist = 'c2.tif' #c2 for G channel
# channel_assist2 = 'c3.tif'  # c3 for R channel, you can also choose c3(R) channel as assist channel 
channel = 'c1.tif' #c1 for B channel,
dir_sou = 'datasets/data_ovarian/{}_input_rename/'.format(train_test) #source assist channel data path after rename.
dir_sou_B = 'datasets/data_ovarian/{}_input_B_rename/'.format(train_test) #source B channel data path after rename.
dir_tar = 'datasets/data_ovarian/{}_GT_rename/'.format(train_test) #target assist channel data path after rename.
imgs_s = glob.glob(os.path.join(dir_sou,'*tif'))
for img_s in imgs_s:
    img_t = img_s.replace('input_rename', 'GT_rename')
    img_s_b = img_s.replace('input_rename_for_B', 'input_B_rename').replace(channel_assist,channel).replace(channel_assist,channel)
    img1 = cv2.imread(img_t)
    img2 = cv2.imread(img_s)
    img2_b = cv2.imread(img_s_b)
    rows1, cols1 = img1.shape[0:2]
    img1 = cv2.resize(img1,(cols1//4,rows1//4))
    detector = cv2.ORB_create(10000,1,1)

    kp1 = detector.detect(img1, None)
    kp2 = detector.detect(img2, None)
    kp1, des1 = detector.compute(img1, kp1)
    kp2, des2 = detector.compute(img2, kp2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    img_name = os.path.basename(img_s_b)
    img3 = drawMatches(img1, kp1, img2_b, kp2, matches[:2000],train_test,img_name)