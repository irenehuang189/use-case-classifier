from flask import Flask, render_template, request, flash, session, redirect, url_for
from datetime import datetime
from werkzeug.utils import secure_filename
import os, sys
import classifier


UPLOAD_FOLDER = '/uploaded_img/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'svg', 'bmp'])

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'mau upload aja susahh'
app.config['SESSION_TYPE'] = 'filesystem'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('file[]')
    return handle_files(files)


def handle_files(files):
    # check if the post request has the file part
    path = os.path.join('static/uploaded_img', datetime.strftime(datetime.now(), '%d%m%Y %H%M%S'))
    if not os.path.exists(path):
        os.makedirs(path)

    print('Hello world!', file=sys.stderr)
    print(files, file=sys.stderr)
    filenames = []
    for file in files:
        if file.filename == '':
            flash('No file selected')
        elif file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(path, filename))
            filenames.append(filename)
        else:
            flash('File extension for ' + file.filename + ' is not allowed')
    (file_paths, features, (image_classes, probabilities)) = classifier.extract_images(path)

    return render_template('result.html', 
                            file_paths = file_paths, features = features, image_classes = image_classes, probabilities = probabilities)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.after_request
def apply_caching(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response


if __name__ == '__main__':
    import pdb; pdb.set_trace()
    app.run(debug = True)
