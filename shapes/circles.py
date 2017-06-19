from shapes import Shapes, Ellipses
import numpy as np
import cv2


class Circles(Shapes):
    circles = []

    def detect_circle(self, contour):
        """Find circle from a contour
            If shape is not found, return none and unidentified string"""
        ellipse = cv2.fitEllipse(contour)
        a, b = Ellipses.get_length(ellipse)
        ellipse_area = np.pi * a * b
        contour_area = cv2.contourArea(contour)
        area_diff = abs(contour_area - ellipse_area)

        if (a != b) or (area_diff > (contour_area*Shapes.MAX_AREA_DIFF_PCT)):
            return None, Shapes.UNIDENTIFIED_SHAPE
        return ellipse, Shapes.CIRCLE_SHAPE

    def draw(self, img, idx=None):
        """Draw circles on image"""
        if idx is None:
            for circle in self.circles:
                self.__draw_circle(img, circle)
        else:
            self.__draw_circle(img, self.circles[idx])

    @staticmethod
    def __draw_circle(img, circle):
        """Draw idx-th circle on image"""
        cv2.ellipse(img, circle, color=(0, 255, 0), thickness=2)
        return img

    @staticmethod
    def get_radius(circle):
        r = circle[1][0] / 2
        return r
