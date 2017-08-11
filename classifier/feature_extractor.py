import numpy as np
import sys

def extract_features(img, lines, triangles, rectangles, rhombuses, ellipses, circles):
    features = np.zeros((1, 8))
    features[0][0] = ellipse_area_pct(ellipses, img)
    features[0][1] = circle_area_pct(circles, img)
    features[0][2] = rectangle_area_pct(rectangles, img)
    features[0][3] = is_triangle_existed(triangles)
    # features[0][4] = is_rhombuses_existed(rhombuses)
    features[0][4] = line_num(ellipses, lines, circles)
    features[0][5] = ellipse_size_sd(ellipses)
    features[0][6] = circle_size_sd(circles)
    features[0][7] = ellipse_horizontal_orientation(ellipses)
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


def line_num(ellipses, lines, circles):
    """F6: Calculate line number in image"""
    # TODO: detect lines connection, not just by assumption that all lines will connect with ellipses and circles
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


def get_arff_header():
    header = ['@relation use_case',
            '@attribute f1 numeric',
            '@attribute f2 numeric',
            '@attribute f3 numeric',
            '@attribute f4 numeric',
            # '@attribute f5 numeric',
            '@attribute f6 numeric',
            '@attribute f7 numeric',
            '@attribute f8 numeric',
            '@attribute f9 numeric',
            '@attribute class {positive, negative}',
            '@data']
    return header
