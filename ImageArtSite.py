from pathlib import Path
import tempfile
import numpy as np
import cv2
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template, session, g
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.secret_key = 'neskirneh'
temp_dir = tempfile.TemporaryDirectory()
app.config['UPLOAD_FOLDER'] = Path(temp_dir.name)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Dont accept posts above 16 MB.
app.config['IMAGE_WH'] = (640, 480)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # For development cache is disable


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            uploaded_file_path = app.config['UPLOAD_FOLDER'] / filename
            file.save(str(uploaded_file_path))
            process_image(uploaded_file_path)
            return redirect(url_for('show_uploaded_file', filename=filename))
        flash("Must be image file (png, jpg, jpeg)")
    return render_template('base.html', file_paths=None)


@app.route('/images/<filename>')
def serve_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)  # , cache_timeout=0


@app.route('/uploads/<filename>')
def show_uploaded_file(filename):
    file_paths = {"uploaded_file_path": url_for('serve_file', filename=filename)}
    return render_template('base.html', file_paths=file_paths)


def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def process_image(uploaded_file_path):
    uploaded_file_path = str(uploaded_file_path)
    user_image = cv2.imread(uploaded_file_path)
    user_image = expand_to_aspect_ratio(user_image, final_shape=app.config['IMAGE_WH'][::-1])
    cv2.imwrite(uploaded_file_path, user_image)
    print(uploaded_file_path)


def expand_to_aspect_ratio(image, aspect_ratio=4/3, final_shape=None):
    if final_shape is not None:
        aspect_ratio = final_shape[1] / final_shape[0]

    h, w = image.shape[:2]

    if h * aspect_ratio > w:
        w_new = round(h * aspect_ratio)
        dw = w_new - w
        dw1 = dw // 2
        image = cv2.copyMakeBorder(image, 0, 0, dw1, dw - dw1, cv2.BORDER_CONSTANT, 0)
    elif w / aspect_ratio > h:
        h_new = round(w / aspect_ratio)
        dh = h_new - h
        dh1 = dh // 2
        image = cv2.copyMakeBorder(image, dh1, dh - dh1, 0, 0, cv2.BORDER_CONSTANT, 0)

    if final_shape:
        image = cv2.resize(image, final_shape[::-1])

    return image


def style_transfer(image):

    models =
    st_images = []
    means_bgr = (103.939, 116.779, 123.680)
    for model_path in models:
        (h, w) = image.shape[:2]
        net = cv2.dnn.readNetFromTorch(model_path)
        blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), means_bgr, swapRB=False, crop=False)
        net.setInput(blob)
        output = net.forward()
        output = output.reshape((3, output.shape[2], output.shape[3]))
        for i in range(3):
            output[i] += means_bgr[i]
        output /= 255.0
        output = output.transpose(1, 2, 0)
        st_images.append(output)

    return st_images
    # Reference: https://www.pyimagesearch.com/2018/08/27/neural-style-transfer-with-opencv/

# Run with cli in ImageArtSite.py folder:
# > export FLASK_APP=ImageArtSite.py			# On windows export -> set
# > export FLASK_ENV=development		        # Use this to enable debug mode. The server will auto reload on code changes.
# > flask run
