import shapes
import cv2


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

    triangles, square_rects, rhombuses, ellipses, circles, other_polygons = ([] for i in range(6))
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
            elif shape_name == shapes.Shapes.SQUARE_RECT_SHAPE:
                square_rects.append(shape)
            elif shape_name == shapes.Shapes.RHOMBUS_SHAPE:
                rhombuses.append(shape)
            elif shape_name == shapes.Shapes.ELLIPSE_SHAPE:
                ellipses.append(shape)
            elif shape_name == shapes.Shapes.CIRCLE_SHAPE:
                circles.append(shape)

    print('Small contour: ', small_contour_cnt)
    print('Triangles: ', len(triangles))
    print('Square_rects: ', len(square_rects))
    print('Rhombuses: ', len(rhombuses))
    print('Ellipses: ', len(ellipses))
    print('Circles: ', len(circles))
    print('Others: ', len(other_polygons))
    return triangles, square_rects, rhombuses, ellipses, circles


def detect_shape(contour):
    """Detect shape in a contour"""
    epsilon = shapes.Shapes.PERIMETER_PCT * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    shape, shape_name = None, shapes.Shapes.UNIDENTIFIED_SHAPE
    if len(approx) == 3:
        shape, shape_name = shapes.Triangles.detect_triangle(contour)
    elif len(approx) == 4:
        shape, shape_name = shapes.SquareRects.detect_square_rect(contour, approx)
        if shape is None:
            shape, shape_name = shapes.Rhombuses.detect_rhombus(approx)
    elif len(approx) >= 4:
        shape, shape_name = shapes.Ellipses.detect_ellipse(contour)
        if shape is None:
            shape, shape_name = shapes.Circles.detect_circle(contour)
    return shape, shape_name
