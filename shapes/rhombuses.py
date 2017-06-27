import cv2
import numpy as np

from shapes.shapes import Shapes


class Rhombuses(Shapes):
    MAX_SIDE_LENGTH_DIFF = 40
    rhombuses = []

    def __init__(self, rhombuses):
        self.rhombuses = rhombuses

    @staticmethod
    def detect_rhombus(approx):
        """Find rhombus from a contour
            If shape is not found, return none and unidentified string"""
        max_length_diff = Rhombuses.get_max_length_diff_in_quad(approx)
        if max_length_diff > Rhombuses.MAX_SIDE_LENGTH_DIFF:
            return None, Shapes.UNIDENTIFIED_SHAPE
        return approx, Shapes.RHOMBUS_SHAPE

    def draw(self, img, idx=None):
        """Draw rhombuses on image"""
        if idx is None:
            for rhombus in self.rhombuses:
                img = self.__draw_rhombus(img, rhombus)
        else:
            img = self.__draw_rhombus(img, self.rhombuses[idx])

        return img

    @staticmethod
    def __draw_rhombus(img, rhombus):
        """Draw idx-th rhombus on image"""
        for i, point in enumerate(rhombus):
            p1 = tuple(rhombus[i][0])
            p2 = tuple(rhombus[(i+1) % 4][0])
            cv2.line(img, p1, p2, color=(29, 131, 255), thickness=2)
        return img

    @staticmethod
    def get_max_length_diff_in_quad(points):
        """Find leftmost, rightmost, uppermost, and bottommost point of a quadrilateral
        and count maximum length between points as a rhombus"""
        leftmost, uppermost, rightmost, bottommost = (points[0, 0] for i in range(4))
        for point in points:
            x = point[0, 0]
            y = point[0, 1]
            if x < leftmost[0]:
                # Point is located on the left side of leftmost point
                leftmost = point[0]
            elif x > rightmost[0]:
                rightmost = point[0]
            elif y < uppermost[1]:
                uppermost = point[0]
            elif y > bottommost[1]:
                bottommost = point[0]

        length_diff = [cv2.norm(uppermost - leftmost),
                       cv2.norm(rightmost - uppermost),
                       cv2.norm(bottommost - rightmost),
                       cv2.norm(leftmost - bottommost)]
        return np.max(length_diff)

    def is_empty(self):
        return len(self.rhombuses) == 0
