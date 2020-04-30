import cv2
import numpy as np


src_img = cv2.imread('./img/1.png')

h, w, _ = src_img.shape
size = max(src_img.shape)

rate = 0.5

img = cv2.resize(src_img, (int(w*rate), int(h*rate)))

img = cv2.Canny(img, 128, 255)

img = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))

contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 切成小图
marks = []
index = 0
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    x = int(x * 2)
    y = int(y * 2)
    w = int(w * 2)
    h = int(h * 2)
    # 小于一定值的就不要了
    # if w < 20 or h < 20:
    #     continue
    index += 1
    marks.append((x, y, w, h))
    part = src_img[y:y+h, x:x+w]
    partpath = './img/part_{}.jpg'.format(index)
    cv2.imwrite(partpath, part)


cv2.imshow('canny', img)
cv2.waitKey(0)
