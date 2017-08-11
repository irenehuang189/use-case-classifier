import cv2
import numpy as np

from shapes.shapes import Shapes


class Circles(Shapes):
    circles = []

    def __init__(self, circles):
        self.circles = circles

    @staticmethod
    def detect_circle(contour):
        """Find circle from a contour
            If shape is not found, return none and unidentified string"""
        ellipse = cv2.fitEllipse(contour)
        a = ellipse[1][0] / 2
        b = ellipse[1][1] / 2
        ellipse_area = np.pi * a * b
        contour_area = cv2.contourArea(contour)
        area_diff = abs(contour_area - ellipse_area)

        if not(Circles.is_circle(a, b)) or (area_diff > (contour_area*Shapes.MAX_AREA_DIFF_PCT)):
            return None, Shapes.UNIDENTIFIED_SHAPE
        return ellipse, Shapes.CIRCLE_SHAPE

    def draw(self, img, idx=None):
        """Draw circles on image"""
        if idx is None:
            for circle in self.circles:
                img = self.__draw_circle(img, circle)
        else:
            img = self.__draw_circle(img, self.circles[idx])

        return img

    @staticmethod
    def __draw_circle(img, circle):
        """Draw idx-th circle on image"""
        # TODO: Fix parameter warning in cv2.ellipse
        cv2.ellipse(img, circle, color=(255, 0, 0), thickness=2)
        return img

    def standard_deviation(self):
        """Count standard deviation of all circles"""
        areas = []
        for circle in self.circles:
            areas.append(Circles.get_area(circle))
        areas = np.array(areas)
        return np.std(areas)

    def average(self):
        return self.total_area() / self.size()

    def total_area(self):
        total_area = 0
        for circle in self.circles:
            total_area += Circles.get_area(circle)
        return total_area

    def size(self):
        return len(self.circles)

    def is_empty(self):
        return len(self.circles) == 0

    @staticmethod
    def get_area(circle):
        r = Circles.get_radius(circle)
        return np.pi * r * r

    @staticmethod
    def get_radius(circle):
        r = circle[1][0] / 2
        return r

    @staticmethod
    def is_circle(length_a, length_b):
        radius_diff = abs(length_a - length_b)
        radius_avg = (length_a + length_b) / 2
        return radius_diff < (radius_avg*Shapes.MAX_RADIUS_DIFF_PCT)
