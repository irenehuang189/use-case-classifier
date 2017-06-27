import cv2

import shapes


def process_image(gray_img):
    threshold_img = threshold_image(gray_img)
    contours = find_contours(threshold_img)

    triangles, rectangles, rhombuses, ellipses, circles = detect_shapes(contours)
    lines = shapes.Lines()
    lines.detect(threshold_img)
    return lines, triangles, rectangles, rhombuses, ellipses, circles

def threshold_image(gray_img):
    _, threshold_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    threshold_img = cv2.bitwise_not(threshold_img)
    return threshold_img


def detect_edge(img):
    otsu_threshold, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high_threshold = otsu_threshold
    low_threshold = 0.5 * otsu_threshold
    edges = cv2.Canny(img, low_threshold, high_threshold)
    return edges


def find_contours(img):
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def detect_shapes(contours):
    """Detect shapes from image contours"""
    print('Contours size:', len(contours))
    small_contour_cnt = 0

    triangles, rectangles, rhombuses, ellipses, circles, other_polygons = ([] for i in range(6))
    for i, contour in enumerate(contours):
        # Area validation
        area = cv2.contourArea(contour)
        if area < shapes.Shapes.MIN_CONTOUR_AREA:
            small_contour_cnt += 1
            continue

        # Detect shape
        shape, shape_name = detect_shape(contour)
        if shape is None:
            other_polygons.append(shape)
        else:
            if shape_name == shapes.Shapes.TRIANGLE_SHAPE:
                triangles.append(shape)
            elif shape_name == shapes.Shapes.RECTANGLE_SHAPE:
                rectangles.append(shape)
            elif shape_name == shapes.Shapes.RHOMBUS_SHAPE:
                rhombuses.append(shape)
            elif shape_name == shapes.Shapes.ELLIPSE_SHAPE:
                ellipses.append(shape)
            elif shape_name == shapes.Shapes.CIRCLE_SHAPE:
                circles.append(shape)

    print('Small contour: ', small_contour_cnt)
    print('Triangles: ', len(triangles))
    print('Rectangles: ', len(rectangles))
    print('Rhombuses: ', len(rhombuses))
    print('Ellipses: ', len(ellipses))
    print('Circles: ', len(circles))
    print('Others: ', len(other_polygons))

    triangles = shapes.Triangles(triangles)
    rectangles = shapes.Rectangles(rectangles)
    rhombuses = shapes.Rhombuses(rhombuses)
    ellipses = shapes.Ellipses(ellipses)
    circles = shapes.Circles(circles)
    return triangles, rectangles, rhombuses, ellipses, circles


def detect_shape(contour):
    """Detect shape in a contour"""
    epsilon = shapes.Shapes.PERIMETER_PCT * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    shape, shape_name = None, shapes.Shapes.UNIDENTIFIED_SHAPE
    if len(approx) == 3:
        shape, shape_name = shapes.Triangles.detect_triangle(contour)
    elif len(approx) == 4:
        shape, shape_name = shapes.Rectangles.detect_rectangle(contour, approx)
        if shape is None:
            shape, shape_name = shapes.Rhombuses.detect_rhombus(approx)
    elif len(approx) >= 4:
        shape, shape_name = shapes.Ellipses.detect_ellipse(contour)
        if shape is None:
            shape, shape_name = shapes.Circles.detect_circle(contour)
    return shape, shape_name


def draw_shapes(img, lines=None, triangles=None, rectangles=None, rhombuses=None, ellipses=None, circles=None):
    if not(lines is None):
        img = lines.draw(img)
    if not(triangles is None):
        img = triangles.draw(img)
    if not(rectangles is None):
        img = rectangles.draw(img)
    if not(rhombuses is None):
        img = rhombuses.draw(img)
    if not(ellipses is None):
        img = ellipses.draw(img)
    if not(circles is None):
        img = circles.draw(img)

    return img
