class Triangles:
    def find(contour):
        _, triangle = cv2.minEnclosingTriangle(contour)
        triangle_area = cv2.contourArea(triangle)
        contour_area = cv2.contourArea(contour)
        area_diff = abs(contour_area - triangle_area)

        if area_diff > (contour_area*MAX_AREA_DIFF_PCT):
            return None, UNIDENTIFIED_SHAPE
        return triangle, TRIANGLE_SHAPE