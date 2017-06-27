import os


def rename_images(path):
    for i, file in enumerate(os.listdir(path)):
        file_ext = os.path.splitext(file)[1].lower()
        print(os.path.join(path, str(i) + file_ext))
        os.rename(os.path.join(path, file), os.path.join(path, str(i) + file_ext))


def convert_images(path):
    not_converted_ext = ('.png', 'PNG', '.jpg', 'JPG', 'jpeg')
    for file in os.listdir(path):
        if not(file.endswith(not_converted_ext)):
            file_name = os.path.splitext(file)[0]
            convert_args = 'magick convert ' + os.path.join(path, file) + ' ' + os.path.join(path, file_name) + '.jpg'
            print(convert_args)
            os.system(convert_args)
            os.remove(os.path.join(path, file))
