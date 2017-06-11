import numpy as np
import cv2
import time

MIN_AREA_OF_CONTOUR = 100
MAX_AREA_PERCENTAGE_ERROR = 0.1

def draw_contours(contours, img):
    """Draw contours to image"""
    for i, contour in enumerate(contours):
        moments = cv2.moments(contour)
        c_x, c_y = 0, 0
        if moments['m00'] != 0:
            c_x = int((moments['m10'] / moments['m00']))
            c_y = int((moments['m01'] / moments['m00']))

        cv2.drawContours(img, [contour], -1, (0, 0, 255), 1)
        cv2.putText(img, str(i), (c_x, c_y), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4, color=(0, 125, 255))
    return img

def detect_shape(contour):
    """Detect shape in a contour"""
    shape = 'unidentified'
    epsilon = 0.1 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 3:
        shape = "triangle"

    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)

        shape = "square" if 0.95 >= ar <= 1.05 else "rectangle"

    elif len(approx) == 5:
        shape = "pentagon"

    else:
        shape = "circle"
    return shape

def detect_shapes(contours, img):
    """Detect shapes from image contours"""
    print('Contours size:', len(contours))
    for i, contour in enumerate(contours):
        # Area validation
        area = cv2.contourArea(contour)
        if area < MIN_AREA_OF_CONTOUR:
            continue

        # Count centroid
        moments = cv2.moments(contour)
        c_x, c_y = 0, 0
        if moments['m00'] != 0:
            c_x = int((moments['m10'] / moments['m00']))
            c_y = int((moments['m01'] / moments['m00']))

        # Detect shape
        shape = detect_shape(contour)
        cv2.drawContours(img, [contour], -1, (0, 0, 255), 1)
        cv2.putText(img, shape, (c_x, c_y), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4, color=(0, 125, 255))
    return img


# Read image
image_name = 'test5.png'
colored_img = cv2.imread(image_name)
gray_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2GRAY)

# Threshold image
_, threshold_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
# cv2.imshow('Threshold', threshold_img)

# Detect edge
retval, _ = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
high_threshold = retval
low_threshold = 0.5 * retval
edges = cv2.Canny(gray_img, low_threshold, high_threshold)
# cv2.imshow('Edges', edges)

# Find contours
_, contours, hierarchy = cv2.findContours(threshold_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_img = draw_contours(contours, np.zeros(colored_img.shape))
cv2.namedWindow('Contours', cv2.WINDOW_NORMAL)
cv2.imshow('Contours', contours_img)
# cv2.imwrite('contours.png', contours_img)

_, canny_contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
canny_contours_img = draw_contours(canny_contours, np.zeros(colored_img.shape))
cv2.namedWindow('Canny Contours', cv2.WINDOW_NORMAL)
cv2.imshow('Canny Contours', canny_contours_img)
# cv2.imwrite('canny_contours.png', canny_contours_img)

# Detect shapes
shapes_img = detect_shapes(contours, np.zeros(colored_img.shape))
cv2.namedWindow('Shapes', cv2.WINDOW_NORMAL)
cv2.imshow('Shapes', shapes_img)

canny_shapes_img = detect_shapes(canny_contours, np.zeros(colored_img.shape))
cv2.namedWindow('Canny Shapes', cv2.WINDOW_NORMAL)
cv2.imshow('Canny Shapes', canny_shapes_img)

cv2.waitKey(0)
cv2.destroyAllWindows()