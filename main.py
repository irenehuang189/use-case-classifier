import numpy as np
import cv2
import math

MIN_CONTOUR_AREA = 100
MAX_AREA_DIFF_PCT = 0.1
PERIMETER_PCT = 0.01
UNIDENTIFIED_SHAPE = 'unidentified'
TRIANGLE_SHAPE = 'triangle'
SQUARE_RECT_SHAPE = 'square rect'
RHOMBUS_SHAPE = 'rhombus'
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

        cv2.drawContours(img, [contour], -1, (255, 255, 255), 1)
        cv2.putText(img, str(i), (c_x, c_y), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4, color=(0, 10, 255))
    return img


def detect_shapes(contours, img):
    """Detect shapes from image contours"""
    print('Contours size:', len(contours))
    cnt_not_convex = 0
    triangles, squarerects, rhombuses, other_polygons, lines, ellipses, circles = ([] for i in range(7))
    for i, contour in enumerate(contours):
        # Area validation
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            cnt_not_convex += 1
            continue

        # Detect shape
        shape, shape_name, color = detect_shape(contour)
        if not(shape is None):
            if shape_name == TRIANGLE_SHAPE:
                triangles.append(shape)
                for j, point in enumerate(shape):
                    pt1 = tuple(shape[j][0])
                    pt2 = tuple(shape[(j+1) % 3][0])
                    cv2.line(img, pt1, pt2, (0,255,0), 2)
            elif shape_name == SQUARE_RECT_SHAPE:
                squarerects.append(shape)
                x, y, w, h = shape[0], shape[1], shape[2], shape[3]
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            elif shape_name == RHOMBUS_SHAPE:
                rhombuses.append(shape)
                for j, point in enumerate(shape):
                    pt1 = tuple(shape[j][0])
                    pt2 = tuple(shape[(j+1) % 4][0])
                    cv2.line(img, pt1, pt2, (255,0,0), 2)
            elif shape_name == LINE_SHAPE:
                lines.append(shape)
                cv2.drawContours(img, shape, -1, (0,255,0), 3)
            elif shape_name == ELLIPSE_SHAPE:
                ellipses.append(shape)
                cv2.ellipse(img, shape, (0,255,0), 2)

        else:
            other_polygons.append(shape)

    print('Small contour: ', cnt_not_convex)
    print('Triangles: ', len(triangles))
    print('Squarerects: ', len(squarerects))
    print('Rhombuses: ', len(rhombuses))
    print('Lines: ', len(lines))
    print('Ellipses: ', len(ellipses))
    print('Others: ', len(other_polygons))
    return img


def detect_shape(contour):
    """Detect shape in a contour"""
    epsilon = PERIMETER_PCT * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 2:
        shape, shape_name = approx, LINE_SHAPE
        color = (255, 255, 255)
    elif len(approx) == 3:
        shape, shape_name = get_triangle(contour, approx)
        color = (0, 0, 255)
    elif len(approx) == 4:
        shape, shape_name = get_quad(contour, approx)
        color = (0, 255, 0)
    else:
        shape, shape_name = get_ellipse(contour, approx)
        color = (255, 0, 0)
    return shape, shape_name, color


def get_triangle(contour, approx):
    """Get triangle from a contour"""
    _, triangle = cv2.minEnclosingTriangle(contour)
    triangle_area = cv2.contourArea(triangle)
    contour_area = cv2.contourArea(contour)
    area_diff = abs(contour_area - triangle_area)

    if area_diff > (contour_area*MAX_AREA_DIFF_PCT):
        return None, UNIDENTIFIED_SHAPE
    return triangle, TRIANGLE_SHAPE


def get_quad(contour, approx):
    """Get quadrilateral from a contour"""
    rect = cv2.boundingRect(approx)
    quad_area = rect[2] * rect[3] # width * height
    contour_area = cv2.contourArea(contour)
    area_diff = abs(contour_area - quad_area)

    # TODO: erase. This if condition is for debugging purpose
    # if (quad_area - colored_img.size) < contour_area*MAX_AREA_DIFF_PCT:
    #     return None, UNIDENTIFIED_SHAPE

    if (area_diff < (contour_area*MAX_AREA_DIFF_PCT)) and (max_cos_in_quad(approx) < 0.1):
        return rect, SQUARE_RECT_SHAPE

    # Try to fit rhombus
    max_length_diff = max_length_diff_in_quad(approx)
    if max_length_diff > 40:
        return None, UNIDENTIFIED_SHAPE
    return approx, RHOMBUS_SHAPE


def max_cos_in_quad(contour):
    """Get maximum cos of quadrilateral edges"""
    max_cos = np.max([angle_cos(contour[i], contour[(i + 1) % 4], contour[(i + 2) % 4]) for i in range(4)])
    return max_cos


def angle_cos(p0, p1, p2):
    """Count cosine of an edges"""
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2.T) / np.sqrt(np.dot(d1, d1.T)*np.dot(d2, d2.T)))


def max_length_diff_in_quad(points):
    """Find leftmost, rightmost, uppermost, and bottommost of a quadrilateral
    and find maximum length between points as a rhombus"""
    # Sequence: leftmost, uppermost, rightmost, bottommost
    leftmost, uppermost, rightmost, bottommost = (points[0, 0] for i in range(4))
    for point in points:
        x = point[0, 0]
        y = point[0, 1]
        if x < leftmost[0]:
            # Point is located on the left side of leftmost point
            leftmost = point[0]
        elif x > rightmost[0]:
            rightmost = point[0]
        elif y < uppermost[1]:
            uppermost = point[0]
        elif y > bottommost[1]:
            bottommost = point[0]

    length_diff = [cv2.norm(uppermost - leftmost),
                   cv2.norm(rightmost - uppermost),
                   cv2.norm(bottommost - rightmost),
                   cv2.norm(leftmost - bottommost)]
    return np.max(length_diff)


def get_ellipse(contour, approx):
    """Get ellipse from a contour"""
    ellipse = cv2.fitEllipse(contour)
    a = ellipse[1][0] / 2
    b = ellipse[1][1] / 2
    ellipse_area = np.pi * a * b
    contour_area = cv2.contourArea(contour)
    area_diff = abs(contour_area - ellipse_area)

    if area_diff > (contour_area*MAX_AREA_DIFF_PCT):
        return None, UNIDENTIFIED_SHAPE
    return ellipse, ELLIPSE_SHAPE


def get_centroid(contour):
    """Get centroid point of a contour"""
    moments = cv2.moments(contour)
    c_x, c_y = 0, 0
    if moments['m00'] != 0:
        c_x = int((moments['m10'] / moments['m00']))
        c_y = int((moments['m01'] / moments['m00']))
    return c_x, c_y


def calculate_mode(img):
    """Calculate median of a grayscale image"""
    hist = cv2.calcHist([img], [0], None, [256], [0,256])
    max, pos = 0, -1
    for i, intensity in enumerate(hist):
        if intensity > max:
            max = intensity
            pos = i
    print('Mode: ', pos, max)
    return pos


def find_lines(source, img):
    """Find lines with hough lines and draw the result on image"""
    lines = cv2.HoughLines(source, 1, np.pi/180, 200)
    print('Jumlah HT Lines', len(lines))
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*a)
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*a)

        cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2)
    cv2.namedWindow('Lines', cv2.WINDOW_NORMAL)
    cv2.imshow('HoughLines', img)

# Read image
# image_name = 'test.png'
# image_name = 'test5.png'
image_name = 'non3.jpg'
colored_img = cv2.imread(image_name)
gray_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2GRAY)

# Threshold image
_, threshold_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
threshold_img = cv2.bitwise_not(threshold_img)
cv2.namedWindow('Threshold', cv2.WINDOW_NORMAL)
cv2.imshow('Threshold', threshold_img)

# Detect edge
otsu_threshold, _ = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print('Otsu Theshold: ', otsu_threshold)
high_threshold = otsu_threshold
low_threshold = 0.5 * otsu_threshold
edges = cv2.Canny(gray_img, low_threshold, high_threshold)
cv2.namedWindow('Edges', cv2.WINDOW_NORMAL)
cv2.imshow('Edges', edges)

# Find contours
_, contours, hierarchy = cv2.findContours(threshold_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_img = draw_contours(contours, np.zeros(colored_img.shape))
cv2.namedWindow('Contours', cv2.WINDOW_NORMAL)
cv2.imshow('Contours', contours_img)

# Detect lines
lines = cv2.HoughLinesP(threshold_img, rho=1, theta=np.pi/180, threshold=200,
                        minLineLength=100, maxLineGap=10)
print('Jumlah HT Lines P', len(lines))
cv2.namedWindow('Lines', cv2.WINDOW_NORMAL)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(colored_img, (x1,y1), (x2,y2), (0,0,255), 3)
    cv2.putText(colored_img, 'Garis', (x1, x2), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.4, color=(0, 0, 255))
    # cv2.waitKey(0)
cv2.imshow('Lines', colored_img)

# See contour one by one
# print('Contours size:', len(contours))
# img = colored_img
# drewn_cnt = 0
# cv2.namedWindow('Shapes', cv2.WINDOW_NORMAL)
# for i, contour in enumerate(contours):
#     contour_area = cv2.contourArea(contour)
#     if contour_area < MIN_CONTOUR_AREA:
#         continue
#     epsilon = PERIMETER_PCT * cv2.arcLength(contour, True)
#     approx = cv2.approxPolyDP(contour, epsilon, True)
#
#     # Give text
#     drewn_cnt += 1
#     moments = cv2.moments(contour)
#     c_x, c_y = 0, 0
#     if moments['m00'] != 0:
#         c_x = int((moments['m10'] / moments['m00']))
#         c_y = int((moments['m01'] / moments['m00']))
#     s = ''
#     text = (str(i), ' ', str(len(approx)), ' ', str(contour_area))
#     cv2.putText(img, s.join(text), (c_x, c_y), cv2.FONT_HERSHEY_SIMPLEX,
#                 fontScale=0.4, color=(0, 0, 255))
#
#     cv2.drawContours(img, [contour], -1, (0,255,0), 2)
#     cv2.drawContours(img, [approx], -1, (0,0,255), 1)
#     cv2.imshow('Shapes', img)
#     key = cv2.waitKey(0)
#
# print('Tergambar: ', drewn_cnt)

# Detect shapes
print('---THRESHOLDING---')
shapes_img = detect_shapes(contours, colored_img)
cv2.namedWindow('Shapes', cv2.WINDOW_NORMAL)
cv2.imshow('Shapes', shapes_img)
# cv2.imwrite('Shapes.png', shapes_img)

# print('---CANNY---')
# canny_shapes_img = detect_shapes(canny_contours, colored_img)
# cv2.namedWindow('Canny Shapes', cv2.WINDOW_NORMAL)
# cv2.imshow('Canny Shapes', canny_shapes_img)

cv2.waitKey(0)
cv2.destroyAllWindows()