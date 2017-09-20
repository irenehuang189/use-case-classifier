import numpy as np
import cv2
import os

import feature_extractor as fe
import shape_detector as sd


def rename_images(path):
    """Rename all images in path with incremental number, starting from zero"""
    for root, subdirs, files in os.walk(path):
        for i, file in enumerate(files):
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext:
                print(os.path.join(root, str(i) + file_ext))
                os.rename(os.path.join(root, file), os.path.join(root, str(i) + file_ext))


def convert_images(path):
    """Convert all non-JPG and non-PNG images to JPG format
        Supported image formats: https://www.imagemagick.org/script/formats.php"""
    not_converted_ext = ('.png', 'PNG', '.jpg', 'JPG', 'jpeg', 'JPEG')
    for root, subdirs, files in os.walk(path):
        for file in files:
            file_ext = os.path.splitext(file)[1]
            if file_ext and not(file.endswith(not_converted_ext)):
                file_name = os.path.splitext(file)[0]
                convert_args = 'magick convert ' + os.path.join(root, file) + ' ' + os.path.join(root, file_name) + '.jpg'
                print(convert_args)
                os.system(convert_args)
                os.remove(os.path.join(root, file))


def export_arff(path, features):
    """Export image features to arff file format"""
    data = fe.get_arff_header()
    for feature in features:
        feature_text = ','.join(f for f in feature)
        data.append(feature_text)
    
    file = open(path, 'w')
    file.write('\n'.join(data))
    file.close()


def show_image(window_name, img):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


path = 'D:/TA/use-case-classifier/img2/'
print('---RENAMING IMAGES---')
# rename_images(path)
print('\n---CONVERT IMAGES---')
# convert_images(path)

print('\n---EXTRACT FEATURES---')
features = np.zeros((0, fe.FEATURE_NUM+1))
for root, subdirs, files in os.walk(path):
    result_path = os.path.join(root, 'extraction_result')
    if (len(files) > 0) and (not os.path.exists(result_path)):
        os.makedirs(result_path)
    for file in files:
        file_ext = os.path.splitext(file)[1]
        if file_ext:
            # Read image
            print(os.path.join(root, file))
            image_path = os.path.join(root, file)
            colored_img = cv2.imread(image_path)
            if colored_img is None:
                print('Image can not be read:', image_path)
                continue
            gray_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2GRAY)

            # Process image
            lines, triangles, rectangles, rhombuses, ellipses, circles = sd.process_image(gray_img)
            colored_img = sd.draw_shapes(colored_img, lines, triangles, rectangles, rhombuses, ellipses, circles)
            # cv2.imwrite(os.path.join(result_path, file), colored_img)

            # Extract features
            image_class = os.path.basename(root)
            feature = fe.extract_features(gray_img, lines, triangles, rectangles, rhombuses, ellipses, circles)
            feature = np.append(feature, image_class)
            features = np.concatenate((features, [feature]))
            print()

print('Features:', features.shape)

arff_path = os.path.join(path, 'use_case.arff')
export_arff(arff_path, features)
