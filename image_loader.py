import os


def load_images(path):
    for file in os.listdir(path):
        # if file.endswith('.png'):
        print(os.path.join(path, file))
