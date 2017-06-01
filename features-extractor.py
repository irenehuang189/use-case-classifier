import numpy as np
import cv2

img = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
# edges2 = cv2.Canny(img, 100, 200)
# img = cv2.imread('test.png')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img, 50, 150, apertureSize=3)

# Hough Lines
# lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
# for line in lines:
#     rho, theta = line[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*a)
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*a)
#     cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2)

# Probalistic Hough Lines
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)

cv2.imshow("Image", img)
cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
