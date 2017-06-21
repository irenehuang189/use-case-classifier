from shapes import Shapes
import numpy as np
import cv2


class Rectangles(Shapes):
    """Contain 90 degrees edges parallelogram shapes: rectangles (and also square as its subset)"""
    rectangles = []

    def __init__(self, rectangles):
        self.rectangles = rectangles

    @staticmethod
    def detect_rectangle(contour, approx):
        """Find rectangle from a contour
            If shape is not found, return none and unidentified string"""
        rect = cv2.boundingRect(approx)
        quad_area = rect[2] * rect[3]  # width * height
        contour_area = cv2.contourArea(contour)
        area_diff = abs(contour_area - quad_area)

        # TODO: erase. If condition below is for debugging purpose
        # if (quad_area - colored_img.size) < contour_area*MAX_AREA_DIFF_PCT:
        #     return None, UNIDENTIFIED_SHAPE

        max_coz = Rectangles.max_cos_in_quad(approx)
        if not((area_diff < (contour_area*Shapes.MAX_AREA_DIFF_PCT)) and (max_coz < 0.1)):
            return None, Shapes.UNIDENTIFIED_SHAPE
        return rect, Shapes.RECTANGLE_SHAPE

    @staticmethod
    def max_cos_in_quad(contour):
        """Get maximum cos of quadrilateral edges"""
        max_cos = np.max([Rectangles.angle_cos(contour[i], contour[(i + 1) % 4], contour[(i + 2) % 4])
                          for i in range(4)])
        return max_cos

    def draw(self, img, idx=None):
        """Draw squares and rectangles on image"""
        if idx is None:
            for rectangle in self.rectangles:
                img = self.__draw_rectangle(img, rectangle)
        else:
            img = self.__draw_rectangle(img, self.rectangles[idx])

        return img

    @staticmethod
    def __draw_rectangle(img, rectangle):
        """Draw idx-th square/rectangle on image"""
        x, y, w, h = rectangle
        cv2.rectangle(img, (x, y), (x+w, y+h), color=(255, 0, 0), thickness=2)
        return img

    @staticmethod
    def get_area(rectangle):
        _, _, w, h = rectangle
        return w * h

    @staticmethod
    def angle_cos(p0, p1, p2):
        """Count cosine of an edge"""
        d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
        return abs(np.dot(d1, d2.T) / np.sqrt(np.dot(d1, d1.T)*np.dot(d2, d2.T)))

    def total_area(self):
        total_area = 0
        for rectangle in self.rectangles:
            total_area += Rectangles.get_area(rectangle)
        return total_area
