import numpy as np
import sys

FEATURE_NUM = 15

def extract_features(img, lines, triangles, rectangles, rhombuses, ellipses, circles):
    features = np.zeros((1, FEATURE_NUM))
    features[0][0] = ellipse_area_pct(ellipses, img)
    # features[0][1] = circle_area_pct(circles, img)
    features[0][1] = rectangle_area_pct(rectangles, img)
    features[0][2] = is_triangle_existed(triangles)
    features[0][3] = is_rhombuses_existed(rhombuses)
    features[0][4] = line_num(lines, triangles, rectangles, rhombuses, ellipses, circles)
    features[0][5] = ellipse_size_sd(ellipses)
    features[0][6] = circle_size_sd(circles)
    features[0][7] = ellipse_horizontal_orientation(ellipses)
    features[0][8] = triangle_area_pct(triangles, img)
    features[0][9] = rhombus_area_pct(rhombuses, img)
    features[0][10], features[0][11], _ = circle_distribution(circles, img)
    features[0][12], features[0][13], features[0][14] = ellipse_distribution(ellipses, img)
    return features


def ellipse_area_pct(ellipses, img):
    """F1: Count percentage area of ellipses on an image compared with image size"""
    return ellipses.total_area() / img.size


def circle_area_pct(circles, img):
    """F2: Count percentage area of circles on an image compared with image size"""
    return circles.total_area() / img.size


def rectangle_area_pct(rectangles, img):
    """F3: Count percentage area of rectangles on an image compared with image size"""
    return rectangles.total_area() / img.size


def is_triangle_existed(triangles):
    """F4: Check appearance of triangles in image"""
    return not(triangles.is_empty())


def is_rhombuses_existed(rhombuses):
    """F5: Check appearance of rhombuses in image"""
    return not(rhombuses.is_empty())


def line_num(lines, triangles, rectangles, rhombuses, ellipses, circles):
    """F6: Calculate line number in image"""
    shapes_num = triangles.size() + rectangles.size() + rhombuses.size() + ellipses.size() + circles.size()
    if shapes_num:
        return lines.size() / shapes_num
    else:
        return lines.size()


def ellipse_size_sd(ellipses):
    """F7: Calculate ellipse size standard deviation divided by ellipse size average in image"""
    if(ellipses.is_empty()):
        return 0
    return ellipses.standard_deviation()


def circle_size_sd(circles):
    """F8: Calculate circle size variance divided by circle size average in image"""
    if(circles.is_empty()):
        return 0
    return circles.standard_deviation()


def ellipse_horizontal_orientation(ellipses):
    """F9: Count number of ellipses with horizontal orientation
    divided by number of ellipses in image"""
    if(ellipses.is_empty()):
        return 0
    return ellipses.horizontal_num() / ellipses.size()


def triangle_area_pct(triangles, img):
    return triangles.total_area() / img.size


def rhombus_area_pct(rhombuses, img):
    return rhombuses.total_area() / img.size


def circle_distribution(circles, img):
    height, width = img.shape
    ratio = width / 3
    circle_in_segment = [0, 0, 0]
    if circles.size() == 0:
        return circle_in_segment

    for circle in circles.circles:
        origin_x = circle[0][0]
        if(origin_x < ratio):
            circle_in_segment[0] += 1
        elif(origin_x < 2*ratio):
            circle_in_segment[1] += 1
        else:
            circle_in_segment[2] += 1

    # for i, circle_num in enumerate(circle_in_segment):
    #     circle_in_segment[i] = circle_num / circles.size()
    return circle_in_segment


def ellipse_distribution(ellipses, img):
    height, width = img.shape
    ratio = width / 3
    ellipse_in_segment = [0, 0, 0]
    if ellipses.size() == 0:
        return ellipse_in_segment

    for ellipse in ellipses.ellipses:
        origin_x = ellipse[0][0]
        if(origin_x < ratio):
            ellipse_in_segment[0] += 1
        elif(origin_x < 2*ratio):
            ellipse_in_segment[1] += 1
        else:
            ellipse_in_segment[2] += 1

    # for i, ellipse_num in enumerate(ellipse_in_segment):
    #     ellipse_in_segment[i] = ellipse_num / ellipses.size()
    return ellipse_in_segment


def get_arff_header():
    header = ['@relation use_case']
    for i in range(1, 18):
        if(i==2) or (i==14):
            continue
        attribute = '@attribute f' + str(i) + ' numeric'
        header.append(attribute)
    header.extend(('@attribute class {positive, negative}',
            '@data'))
    return header
