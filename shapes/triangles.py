from shapes import Shapes
import cv2


class Triangles(Shapes):
    triangles = []

    def __init__(self, triangles):
        self.triangles = triangles

    @staticmethod
    def detect_triangle(contour):
        """Find triangle from a contour
            If shape is not found, return none and unidentified string"""
        _, triangle = cv2.minEnclosingTriangle(contour)
        triangle_area = cv2.contourArea(triangle)
        contour_area = cv2.contourArea(contour)
        area_diff = abs(contour_area - triangle_area)

        if area_diff > (contour_area * Shapes.MAX_AREA_DIFF_PCT):
            return None, Shapes.UNIDENTIFIED_SHAPE
        return triangle, Shapes.TRIANGLE_SHAPE

    def draw(self, img, idx=None):
        """Draw triangles on image"""
        if idx is None:
            print('Triangles', self.triangles)
            for triangle in self.triangles:
                img = self.__draw_triangle(img, triangle)
        else:
            img = self.__draw_triangle(img, self.triangles[idx])

        return img

    @staticmethod
    def __draw_triangle(img, triangle):
        """Draw idx-th triangles on image"""
        for i, point in enumerate(triangle):
            p1 = tuple(triangle[i][0])
            p2 = tuple(triangle[(i+1) % 3][0])
            cv2.line(img, p1, p2, color=(255, 255, 0), thickness=2)
        return img
