import numpy as np
import cv2
# from grade_mode import grade_mode
import glob
import os

train_test = 'test'

def drawMatches(img1, kp1, img2, kp2, matches, img_name):
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    warpImg_rgb = np.zeros([rows2, cols2])
    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
    # stitch image
    out[:rows1, :cols1] = img1
    out[:rows2, cols1:] = img2
    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt
        # draw map
        cv2.circle(out, (int(x1), int(y1)), 4, (255, 255, 0), 1)
        cv2.circle(out, (int(x2) + cols1, int(y2)), 4, (0, 255, 255), 1)
        cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (255, 0, 0), 1)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0)
    warpImg = cv2.warpPerspective(img2, np.array(M), (img2.shape[1], img2.shape[0]),
                                  flags=cv2.WARP_INVERSE_MAP)
    img_path = 'datasets/data_ovarian/{}_input_rename_warped/'.format(train_test)#save warped source data path.
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    cv2.imwrite(img_path+img_name, warpImg)
    return out

if __name__ == '__main__':
    dir_sou = 'datasets/data_ovarian/{}_input_rename/'.format(train_test) #source data path after rename.
    dir_tar = 'datasets/data_ovarian/{}_GT_rename/'.format(train_test) #target data path after rename. 
    imgs_s = glob.glob(os.path.join(dir_sou,'*tif'))
    for img_s in imgs_s:
        img_t = img_s.replace('input_rename','GT_rename')
        img1 = cv2.imread(img_t)
        img2 = cv2.imread(img_s)
        rows1, cols1 = img1.shape[0:2]
        img1 = cv2.resize(img1,(cols1//4,rows1//4))

        detector = cv2.ORB_create(10000,1,1)

        kp1 = detector.detect(img1, None)
        kp2 = detector.detect(img2, None)
        kp1, des1 = detector.compute(img1, kp1)
        kp2, des2 = detector.compute(img2, kp2)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        img_name = os.path.basename(img_s)
        img3 = drawMatches(img1, kp1, img2, kp2, matches[:2000],img_name)