from shapes import Shapes
import numpy as np
import cv2


class Lines(Shapes):
    lines = []

    def detect(self, img):
        """Detect lines from an image"""
        self.lines = cv2.HoughLinesP(img, rho=1, theta=np.pi/180, threshold=200,
                                     minLineLength=100, maxLineGap=10)

    def draw(self, img, idx=None):
        """Draw lines on image"""
        if idx is None:
            for line in self.lines:
                img = self.__draw_line(img, line)
        else:
            img = self.__draw_line(img, self.lines[idx])

        return img

    @staticmethod
    def __draw_line(img, line):
        """Draw idx-th circle on image"""
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=3)
        return img