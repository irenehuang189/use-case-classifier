import numpy as np
import cv2

imageName = "test2.png"
realImage = cv2.imread(imageName)
img = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)

# Detect edge
# _img, otsuThreshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# highThreshold = otsuThreshold[0][0]
# lowThreshold = otsuThreshold[0][0] * 0.5

highThreshold = 255
lowThreshold = 0.3 * 255
edges = cv2.Canny(img, lowThreshold, highThreshold)

# Find line segments
# lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
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
#     cv2.line(realImage, (x1,y1), (x2,y2), (0,0,255), 2)

lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, 
                        minLineLength=100, maxLineGap=10)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(realImage, (x1,y1), (x2,y2), (255,0,0), 2)

# Detect circles
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                           param1=255, param2=30, minRadius=5, maxRadius=150)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # Draw outer circle
    cv2.circle(realImage, (i[0],i[1]), i[2], (0,255,0), 2)
    # Draw center of the circle
    cv2.circle(realImage, (i[0],i[1]), 2, (0,0,255), 3)

# Simple blob detector
detector = cv2.SimpleBlobDetector_create()
keypoints = detector.detect(img)
im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Keypoints", im_with_keypoints)

# MSER
mser = cv2.MSER_create()
regions, _ = mser.detectRegions(img)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
vis = cv2.imread(imageName)
cv2.polylines(vis, hulls, 1, (255, 0, 0))
cv2.imshow('MSER', vis)

cv2.imshow("Edges", edges)
cv2.imshow("Image", realImage)

cv2.waitKey(0)
cv2.destroyAllWindows()
