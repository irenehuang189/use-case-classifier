from shapes import Shapes
import numpy as np
import cv2


class SquareRects(Shapes):
    """Contain 90 degrees edges parallelogram shapes: squares and rectangles"""
    square_rects = []

    def __init__(self, square_rects):
        self.square_rects = square_rects

    @staticmethod
    def detect_square_rect(contour, approx):
        """Find square/rectangle from a contour
            If shape is not found, return none and unidentified string"""
        rect = cv2.boundingRect(approx)
        quad_area = rect[2] * rect[3]  # width * height
        contour_area = cv2.contourArea(contour)
        area_diff = abs(contour_area - quad_area)

        # TODO: erase. If condition below is for debugging purpose
        # if (quad_area - colored_img.size) < contour_area*MAX_AREA_DIFF_PCT:
        #     return None, UNIDENTIFIED_SHAPE

        max_coz = SquareRects.max_cos_in_quad(approx)
        if not((area_diff < (contour_area*Shapes.MAX_AREA_DIFF_PCT)) and (max_coz < 0.1)):
            return None, Shapes.UNIDENTIFIED_SHAPE
        return rect, Shapes.SQUARE_RECT_SHAPE

    @staticmethod
    def max_cos_in_quad(contour):
        """Get maximum cos of quadrilateral edges"""
        max_cos = np.max([SquareRects.angle_cos(contour[i], contour[(i + 1) % 4], contour[(i + 2) % 4]) for i in range(4)])
        return max_cos

    def draw(self, img, idx=None):
        """Draw squares and rectangles on image"""
        if idx is None:
            for square_rect in self.square_rects:
                self.__draw_square_rect(img, square_rect)
        else:
            self.__draw_square_rect(img, self.square_rects[idx])

    @staticmethod
    def __draw_square_rect(img, square_rect):
        """Draw idx-th square/rectangle on image"""
        x, y, w, h = square_rect
        cv2.rectangle(img, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=2)
        return img

    @staticmethod
    def angle_cos(p0, p1, p2):
        """Count cosine of an edge"""
        d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
        return abs(np.dot(d1, d2.T) / np.sqrt(np.dot(d1, d1.T)*np.dot(d2, d2.T)))
