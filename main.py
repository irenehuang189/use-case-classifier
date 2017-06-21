import cv2
import shape_detection as sd
import shapes


def show_image(window_name, img):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)


def draw_shapes(img, lines=None, triangles=None, square_rects=None, rhombuses=None, circles=None, ellipses=None):
    if not(lines is None):
        img = lines.draw(img)
    if not(triangles is None):
        img = triangles.draw(img)
    if not(square_rects is None):
        img = square_rects.draw(img)
    if not(rhombuses is None):
        img = rhombuses.draw(img)
    if not(circles is None):
        img = circles.draw(img)
    if not(ellipses is None):
        img = ellipses.draw(img)

    return img


# image_name = 'test.png'
image_name = 'test5.png'
# image_name = 'non3.jpg'
colored_img = cv2.imread(image_name)

gray_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2GRAY)
threshold_img = sd.threshold_image(gray_img)
contours = sd.find_contours(threshold_img)

triangles, square_rects, rhombuses, ellipses, circles = sd.detect_shapes(contours)
lines = shapes.Lines()
lines.detect(threshold_img)

triangles = shapes.Triangles(triangles)
square_rects = shapes.SquareRects(square_rects)
rhombuses = shapes.Rhombuses(rhombuses)
circles = shapes.Circles(circles)
ellipses = shapes.Ellipses(ellipses)

draw_shapes(colored_img, lines, triangles, square_rects, rhombuses, circles, ellipses)
show_image('Result', colored_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
