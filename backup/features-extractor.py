import numpy as np
import cv2


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )


def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares

imageName = 'test5.png'
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
cv2.imshow('Keypoints', im_with_keypoints)

# MSER
mser = cv2.MSER_create()
regions, _ = mser.detectRegions(img)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
vis = cv2.imread(imageName)
cv2.polylines(vis, hulls, 1, (255, 0, 0))
cv2.imshow('MSER', vis)

# Find contours
imgContours = cv2.imread(imageName)
im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours, contours, -1, (0, 0, 255), 3)
cv2.imshow('Contours', imgContours)

# Find squares
imgSquares = cv2.imread(imageName)
squares = find_squares(imgSquares)
cv2.drawContours(imgSquares, squares, -1, (0, 0, 255), 3)
cv2.imshow('Squares', imgSquares)

cv2.imshow('Edges', edges)
cv2.imshow('Image', realImage)

cv2.waitKey(0)
cv2.destroyAllWindows()
