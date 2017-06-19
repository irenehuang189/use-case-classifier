from shapes import Shapes
import numpy as np
import cv2


class Ellipses(Shapes):
    ellipses = []

    def detect_ellipse(self, contour):
        """Find ellipse from a contour
            If shape is not found, return none and unidentified string"""
        ellipse = cv2.fitEllipse(contour)
        a, b = self.get_length(ellipse)
        ellipse_area = np.pi * a * b
        contour_area = cv2.contourArea(contour)
        area_diff = abs(contour_area - ellipse_area)

        if (a == b) or (area_diff > (contour_area*Shapes.MAX_AREA_DIFF_PCT)):
            return None, Shapes.UNIDENTIFIED_SHAPE
        return ellipse, Shapes.ELLIPSE_SHAPE

    def draw(self, img, idx=None):
        """Draw ellipses on image"""
        if idx is None:
            for ellipse in self.ellipses:
                self.__draw_ellipse(img, ellipse)
        else:
            self.__draw_ellipse(img, self.ellipses[idx])

    @staticmethod
    def __draw_ellipse(img, ellipse):
        """Draw idx-th ellipse on image"""
        cv2.ellipse(img, ellipse, color=(0, 255, 0), thickness=2)
        return img

    @staticmethod
    def get_length(ellipse):
        a = ellipse[1][0] / 2
        b = ellipse[1][1] / 2
        return a, b
