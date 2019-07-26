from pathlib import Path
import tempfile
import numpy as np
import cv2
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template, session, g
from werkzeug.utils import secure_filename
import click


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
            file.save(filename2path(filename))
            process_upload(filename)
            return redirect(url_for('show_uploaded_file', filename=filename))
        flash("Must be image file (png, jpg, jpeg)")
    return render_template('base.html', file_paths=None)


@app.route('/images/<filename>')
def serve_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/uploads/<filename>')
def show_uploaded_file(filename):
    g.filename = filename
    file_paths = {"uploaded_path": url_for('serve_file', filename=filename)}
    return render_template('style_choose.html', file_paths=file_paths)


@app.route('/uploads/<filename>/<style>')
def show_style_transfer_image(filename, style):
    g.filename = filename
    filename_st = apply_style_transfer(filename, style)
    file_paths = {"uploaded_path": url_for('serve_file', filename=filename),
                  "style_transfer_path": url_for('serve_file', filename=filename_st)}
    return render_template('style_transfer.html', file_paths=file_paths)


@app.cli.command('model-download')
def model_dl_command():
    import subprocess
    subprocess.call(['./static/download_style_transfer_models.sh'])
    click.echo('Downloaded style transfer models.')


def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def filename2path(filename):
    return str(app.config['UPLOAD_FOLDER'] / filename)


def process_upload(filename):
    uploaded_path = filename2path(filename)
    user_image = cv2.imread(uploaded_path)
    user_image = expand_to_aspect_ratio(user_image, final_shape=app.config['IMAGE_WH'][::-1])
    cv2.imwrite(uploaded_path, user_image)


def apply_style_transfer(filename, style):
    uploaded_path = filename2path(filename)
    user_image = cv2.imread(uploaded_path)
    st_path = uploaded_path.rsplit('.', 1)[0] + '_st.png'
    st_filename = Path(st_path).name

    model_dir1 = './static/models/eccv16/'
    model_dir2 = './static/models/instance_norm/'
    style_model_map = \
        {'candy': model_dir2 + 'candy.t7',
         'composition_vii': model_dir1 + 'composition_vii.t7',
         'feathers': model_dir2 + 'feathers.t7',
         'la_muse': model_dir2 + 'la_muse.t7',
         'mosaic': model_dir2 + 'mosaic.t7',
         'starry_night': model_dir1 + 'starry_night.t7',
         'the_scream': model_dir2 + 'the_scream.t7',
         'the_wave': model_dir1 + 'the_wave.t7',
         'udnie': model_dir2 + 'udnie.t7'}
    model_path = style_model_map[style]

    st_image = style_transfer(user_image, model_path)
    cv2.imwrite(st_path, st_image)
    return st_filename


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


def style_transfer(image, model_path):
    means_bgr = (103.939, 116.779, 123.680)
    (h, w) = image.shape[:2]
    net = cv2.dnn.readNetFromTorch(model_path)
    blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), means_bgr, swapRB=False, crop=False)
    net.setInput(blob)
    output = net.forward()
    output = output.reshape((3, output.shape[2], output.shape[3]))
    for i in range(3):
        output[i] += means_bgr[i]
    output = output.transpose(1, 2, 0).clip(min=0, max=255).astype(np.uint8)

    return output
    # Reference: https://www.pyimagesearch.com/2018/08/27/neural-style-transfer-with-opencv/

# Run with cli in ImageArtSite.py folder:
# > export FLASK_APP=ImageArtSite.py			# On windows export -> set
# > export FLASK_ENV=development		        # Use this to enable debug mode. The server will auto reload on code changes.
# > flask model-download
# > flask run
