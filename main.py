import numpy as np
import cv2
import time

MIN_CONTOUR_AREA = 100
MAX_AREA_DIFF_PCT = 0.1
PERIMETER_PCT = 0.1
UNIDENTIFIED_SHAPE = 'unidentified'
TRIANGLE_SHAPE = 'triangle'
SQUARE_RECT_SHAPE = 'square rect'
OTHER_POLYGON_SHAPE = 'other polygon'
LINE_SHAPE = 'line'
CIRCLE_SHAPE = 'circle'
ELLIPSE_SHAPE = 'ellipse'

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


def angle_cos(p0, p1, p2):
    """Count cosine of an edges"""
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1)*np.dot(d2, d2)))


def max_cos_in_quad(contour):
    """Get maximum cos of quadrilateral edges"""
    max_cos = np.max([angle_cos(contour[i], contour[(i + 1) % 4], contour[(i + 2) % 4])
                      for i in range(4)])
    return max_cos


def get_triangle(contour):
    """Get triangle from a contour"""
    _, triangle = cv2.minEnclosingTriangle(contour)
    triangle_area = cv2.contourArea(triangle)
    contour_area = cv2.contourArea(contour)
    diff_area = abs(contour_area - triangle_area)

    if diff_area > (diff_area*MAX_AREA_DIFF_PCT):
        return None, UNIDENTIFIED_SHAPE
    return triangle, TRIANGLE_SHAPE


def is_sides_equal(rect):
    """Detect if a quadrilateral has the same length on all its sides"""



def get_quad(contour):
    """Get quadrilateral from a contour"""
    x, y, w, h = cv2.boundingRect(contour)
    quad_area = w * h
    contour_area = cv2.contourArea(contour)
    diff_area = abs(contour_area - quad_area)

    if diff_area > (diff_area*MAX_AREA_DIFF_PCT) and max_cos_in_quad(contour) < 0.1:
        return (x, y, w, h), SQUARE_RECT_SHAPE

    # Try rotated rectangle
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)


def detect_shape(contour):
    """Detect shape in a contour"""
    shape_name = 'unidentified'
    epsilon = PERIMETER_PCT * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 3:
        shape, shape_name = get_triangle(contour)
        color = (0, 0, 255)
    elif len(approx) == 4:
        if max_cos_in_quad(contour) < 0.1:
            shape_name = 'squarerect'

        color = (0, 255, 0)
    elif len(approx) == 5:
        shape_name = 'pentagon'
        color = (255, 0, 0)
    else:
        if(len(approx)) == 2:
            print('2')
        shape_name = str(len(approx))
        color = (255, 255, 255)
    return shape, shape_name, color


def detect_shapes(contours, img):
    """Detect shapes from image contours"""
    print('Contours size:', len(contours))
    cnt_not_convex = 0
    triangles, squarerects, rhombuses, other_polygons, lines, ellipses, circles = ([] for i in range(5))
    for i, contour in enumerate(contours):
        print('Contour ', i, ':', contour)
        # Area validation
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            cnt_not_convex += 1
            continue

        # Count centroid
        moments = cv2.moments(contour)
        c_x, c_y = 0, 0
        if moments['m00'] != 0:
            c_x = int((moments['m10'] / moments['m00']))
            c_y = int((moments['m01'] / moments['m00']))

        # Detect shape
        shape, shape_name, color = detect_shape(contour)
        if not(shape is None):
            if shape_name == 'triangle':
                triangles.append(object)
            elif shape_name == 'squarerect':

        cv2.drawContours(img, [contour], -1, color, -1)
        cv2.putText(img, shape_name, (c_x, c_y), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4, color=(0, 125, 255))

    print('Not convex cnt: ', cnt_not_convex)
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