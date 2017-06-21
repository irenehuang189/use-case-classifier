from shapes import Shapes
import numpy as np
import cv2


class Ellipses(Shapes):
    ellipses = []

    def __init__(self, ellipses):
        self.ellipses = ellipses

    @staticmethod
    def detect_ellipse(contour):
        """Find ellipse from a contour
            If shape is not found, return none and unidentified string"""
        ellipse = cv2.fitEllipse(contour)
        a, b = Ellipses.get_length(ellipse)
        ellipse_area = np.pi * a * b
        contour_area = cv2.contourArea(contour)
        area_diff = abs(contour_area - ellipse_area)

        if (Ellipses.is_circle(a, b)) or (area_diff > (contour_area*Shapes.MAX_AREA_DIFF_PCT)):
            return None, Shapes.UNIDENTIFIED_SHAPE
        return ellipse, Shapes.ELLIPSE_SHAPE

    def draw(self, img, idx=None):
        """Draw ellipses on image"""
        if idx is None:
            for ellipse in self.ellipses:
                img = self.__draw_ellipse(img, ellipse)
        else:
            img = self.__draw_ellipse(img, self.ellipses[idx])

        return img

    @staticmethod
    def __draw_ellipse(img, ellipse):
        """Draw idx-th ellipse on image"""
        # TODO: Fix parameter warning in cv2.ellipse
        cv2.ellipse(img, ellipse, color=(0, 255, 0), thickness=2)
        return img

    @staticmethod
    def is_circle(length_a, length_b):
        radius_diff = abs(length_a - length_b)
        radius_avg = (length_a + length_b) / 2
        return radius_diff < (radius_avg*Shapes.MAX_RADIUS_DIFF_PCT)

    @staticmethod
    def get_length(ellipse):
        a = ellipse[1][0] / 2
        b = ellipse[1][1] / 2
        return a, b