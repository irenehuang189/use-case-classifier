import numpy as np
import cv2, os, sys, subprocess

import feature_extractor as fe
import shape_detector as sd


def extract_images(path):
    print('---RENAMING IMAGES---', file=sys.stderr)
    rename_images(path)
    print('\n---CONVERT IMAGES---', file=sys.stderr)
    convert_images(path)

    print('\n---EXTRACT FEATURES---', file=sys.stderr)
    file_paths = []
    features = np.zeros((0, 9))
    for root, subdirs, files in os.walk(path):
        result_path = os.path.join(root, 'extraction_result')
        if (len(files) > 0) and (not os.path.exists(result_path)):
            os.makedirs(result_path)
        for file in files:
            file_ext = os.path.splitext(file)[1]
            if file_ext:
                # Read image
                file_paths.append(os.path.join(root, file))
                print(os.path.join(root, file), file=sys.stderr)
                image_path = os.path.join(root, file)
                colored_img = cv2.imread(image_path)
                gray_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2GRAY)

                # Process image
                lines, triangles, rectangles, rhombuses, ellipses, circles = sd.process_image(gray_img)
                colored_img = sd.draw_shapes(colored_img, lines, triangles, rectangles, rhombuses, ellipses, circles)
                cv2.imwrite(os.path.join(result_path, file), colored_img)

                # Extract features
                image_class = '?'
                feature = fe.extract_features(gray_img, lines, triangles, rectangles, rhombuses, ellipses, circles)
                feature = np.append(feature, image_class)
                features = np.concatenate((features, [feature]))
                print('', file=sys.stderr)

    print('Features:' + str(features.shape), file=sys.stderr)

    arff_path = os.path.join(path, 'use_case.arff')
    export_arff(arff_path, features)
    return (file_paths, features, classify_arff(arff_path))


def rename_images(path):
    """Rename all images in path with incremental number, starting from zero"""
    for root, subdirs, files in os.walk(path):
        for i, file in enumerate(files):
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext:
                print(os.path.join(root, str(i) + file_ext), file=sys.stderr)
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
                print(convert_args, file=sys.stderr)
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


def classify_arff(path):
    command = 'java -classpath lib/weka.jar weka.classifiers.trees.RandomForest -T "' + path + '" -l model/1.model -p 0'
    print(command, file=sys.stderr)
    proc = subprocess.Popen(command, stdout = subprocess.PIPE)
    (out, err) = proc.communicate()

    texts = out.decode().split()
    image_classes, probabilities = [], []
    for (i, text) in enumerate(texts):
        if i > 0:
            if ':' in texts[i-1]:
                if ':' in texts[i]:
                    image_class = text.split(':', 1)[1]
                    image_classes.append(image_class)
                else:
                    probabilities.append(text)

    return (image_classes, probabilities)