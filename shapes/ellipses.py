import cv2
import numpy as np

from shapes.shapes import Shapes


class Ellipses(Shapes):
    ellipses = []

    def __init__(self, ellipses):
        self.ellipses = ellipses

    @staticmethod
    def detect_ellipse(contour):
        """Find ellipse from a contour
            If shape is not found, return none and unidentified string"""
        ellipse = cv2.fitEllipse(contour)
        ellipse_area = Ellipses.get_area(ellipse)
        contour_area = cv2.contourArea(contour)
        area_diff = abs(contour_area - ellipse_area)

        length_a, length_b = Ellipses.get_length(ellipse)
        if (Ellipses.is_circle(length_a, length_b)) or (area_diff > (contour_area*Shapes.MAX_AREA_DIFF_PCT)):
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

    def horizontal_num(self):
        """Count number of ellipses with horizontal orientation"""
        count = 0
        for ellipse in self.ellipses:
            length_a, length_b = Ellipses.get_length(ellipse)
            if length_a < length_b:
                count += 1
        return count

    def standard_deviation(self):
        """Count standard deviation of all ellipses"""
        areas = []
        for ellipse in self.ellipses:
            areas.append(Ellipses.get_area(ellipse))
        areas = np.array(areas)
        return np.std(areas)

    def total_area(self):
        total_area = 0
        for ellipse in self.ellipses:
            total_area += Ellipses.get_area(ellipse)
        return total_area

    def size(self):
        return len(self.ellipses)
        
    @staticmethod
    def get_area(ellipse):
        length_a, length_b = Ellipses.get_length(ellipse)
        return np.pi * length_a * length_b

    @staticmethod
    def get_length(ellipse):
        length_a = ellipse[1][0] / 2
        length_b = ellipse[1][1] / 2
        return length_a, length_b

    @staticmethod
    def is_circle(length_a, length_b):
        radius_diff = abs(length_a - length_b)
        radius_avg = (length_a + length_b) / 2
        return radius_diff < (radius_avg*Shapes.MAX_RADIUS_DIFF_PCT)
