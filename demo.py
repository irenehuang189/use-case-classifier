import cv2
import numpy as np

image_name = 'test5.png'
colored_img = cv2.imread(image_name)
gray_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2GRAY)

# Threshold image
_, threshold_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
_, contours, hierarchy = cv2.findContours(threshold_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for i, contour in enumerate(contours):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    print(rect)
    print('Box:', box)

    if i == 2:
        break;
