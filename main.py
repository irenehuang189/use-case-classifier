import numpy as np
import cv2
import os

import feature_extractor as fe
import shape_detector as sd


def rename_images(path):
    for i, file in enumerate(os.listdir(path)):
        file_ext = os.path.splitext(file)[1].lower()
        if file_ext:
            print(os.path.join(path, str(i) + file_ext))
            os.rename(os.path.join(path, file), os.path.join(path, str(i) + file_ext))


def convert_images(path):
    not_converted_ext = ('.png', 'PNG', '.jpg', 'JPG', 'jpeg')
    for file in os.listdir(path):
        file_ext = os.path.splitext(file)[1]
        if file_ext and not(file.endswith(not_converted_ext)):
            file_name = os.path.splitext(file)[0]
            convert_args = 'magick convert ' + os.path.join(path, file) + ' ' + os.path.join(path, file_name) + '.jpg'
            print(convert_args)
            os.system(convert_args)
            os.remove(os.path.join(path, file))


def show_image(window_name, img):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


path = 'D:/TA/usecase-dataset/coba'
result_path = os.path.join(path, 'extraction_result')
print('---RENAMING IMAGES---')
rename_images(path)
print('\n---CONVERT IMAGES---')
convert_images(path)

print('\n---EXTRACT FEATURES---')
if not os.path.exists(result_path):
    os.makedirs(result_path)

features = np.zeros((0, 9))
for file in os.listdir(path):
    file_ext = os.path.splitext(file)[1]
    if file_ext:
        print(os.path.join(result_path, file))
        image_path = os.path.join(path, file)
        colored_img = cv2.imread(image_path)
        gray_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2GRAY)

        lines, triangles, rectangles, rhombuses, ellipses, circles = sd.process_image(gray_img)
        colored_img = sd.draw_shapes(colored_img, lines, triangles, rectangles, rhombuses, ellipses, circles)
        cv2.imwrite(os.path.join(result_path, file), colored_img)

        feature = fe.extract_features(gray_img, lines, triangles, rectangles, rhombuses, ellipses, circles)
        features = np.concatenate((features, feature))
        print()

print('\nFeatures result:')
print(features)
