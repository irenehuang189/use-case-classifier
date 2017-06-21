import cv2
import shape_detection as sd


def show_image(window_name, img):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)


image_name = 'test.png'
# image_name = 'test5.png'
# image_name = 'non3.jpg'
colored_img = cv2.imread(image_name)

gray_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2GRAY)
threshold_img = sd.threshold_image(gray_img)
contours = sd.find_contours(threshold_img)

triangles, square_rects, rhombuses, ellipses, circles = sd.detect_shape(contours)
circles = sd.Circles(circles)
ellipses = sd.Ellipses(ellipses)
lines = sd.Lines()
rhombuses = sd.Rhombuses(rhombuses)
square_rects = sd.SquareRects(square_rects)
triangles = sd.Triangles(triangles)

circles.draw(colored_img)
show_image('Result', colored_img)
