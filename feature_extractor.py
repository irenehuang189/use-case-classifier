def extract_features(img, lines, triangles, rectangles, rhombuses, circles, ellipses):
    ellipse_area_pct(ellipses, img)
    circle_area_pct(circles, img)
    rectangle_area_pct(rectangles, img)
    is_triangle_existed(triangles)
    is_rhombuses_existed(rhombuses)
    ellipse_connection_freq(ellipses, lines, circles)
    ellipse_size_sd(ellipses)
    circle_size_sd(circles)
    ellipse_horizontal_orientation(ellipses)


def ellipse_area_pct(ellipses, img):
    """F1: Count percentage area of ellipses on an image compared with image size"""
    return ellipses.total_area() / img.size


def circle_area_pct(circles, img):
    """F2: Count percentage area of circles on an image compared with image size"""
    print('F2', circles.total_area(), img.size)
    return circles.total_area() / img.size


def rectangle_area_pct(rectangles, img):
    """F3: Count percentage area of rectangles on an image compared with image size"""
    print('F3', rectangles.total_area(), img.size)
    return rectangles.total_area() / img.size


def is_triangle_existed(triangles):
    """F4: Check appearance of triangles in image"""
    return triangles.is_empty()


def is_rhombuses_existed(rhombuses):
    """F5: Check appearance of rhombuses in image"""
    return rhombuses.is_empty()


def ellipse_connection_freq(ellipses, lines, circles):
    """F6: Calculate number of ellipse connected via a line with at least a circle
    divided by total number of ellipse in image"""
    # TODO: detect lines connection, not just by assumption that all lines will connect with ellipses and circles
    return lines.size()


def ellipse_size_sd(ellipses):
    """F7: Calculate ellipse size variance in image"""
    return ellipses.standard_deviation()


def circle_size_sd(circles):
    """F8: Calculate circle size variance in image"""
    return circles.standard_deviation()


def ellipse_horizontal_orientation(ellipses):
    """F9: Count number of ellipses with horizontal orientation
    divided by number of ellipses in image"""
    return ellipses.horizontal_num()
