# Set up Flask: https://www.youtube.com/watch?v=GHvj1ivQ7ms
# Upload a picture: https://www.youtube.com/watch?v=GeiUTkSAJPs
# In VSCODE: CTRL + SHIFT + P => search for 'PYTHON INTERPRETER' => select env
# In Terminal : source env/bin/activate => activate the virtual environment
#  pip install pipreqs
#  pipreqs

# -------------------- IMPORTS ------------------ #
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_wtf import FlaskForm
from wtforms import FileField, StringField, PasswordField, SubmitField, BooleanField, URLField
from wtforms.validators import DataRequired, Length, Email, EqualTo, URL
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
import shutil
# from IPython.display import Image, HTML

# import pickle


# DS
# import numpy as np
import pandas as pd
# NN libraries
import tensorflow as tf
# print(tf.__version__)
from tensorflow import keras
# from keras.applications.vgg16 import VGG16
# from keras.applications.vgg16 import preprocess_input
# from keras.applications.mobilenet_v2 import MobileNetV2

# image processing
# from glob import glob
# from PIL import Image as PIL_Image
from keras import utils
# from keras.preprocessing import image as keras_image
# from keras.preprocessing.image import ImageDataGenerator

# FaceFeatures detection
import cv2
import dlib
import imutils
from PIL import Image
from imutils import face_utils

# Scrapping
from sys import stderr
import requests
from bs4 import BeautifulSoup as bs


# ------------------------------------------ APPLICATION START  ------------------------------------------ #

app = Flask(__name__)

# ---------------------- PARAMETERS -------------------- #
IMG_SIZE = [48, 48, 3]
ALLOWED_SCRAPPED_IMAGES = 30
PICTURES = 'static/pictures/'
UPLOAD_FOLDER = 'static/files/'
UPLOAD_FOLDER_SCRAP = 'static/scrapfiles/'
CROPPED_FOLDER = 'static/cropped/'
CROPPED_FOLDER_SCRAP = 'static/croppedscrap/'
MODELS = 'static/model/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}  # upload image page
labeldic = {'angry': 0, 'disgust': 1, 'fear': 2,
            'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
TLD_list = ['.edu', '.info', '.biz', '.co',
            '.gov', '.mil', '.me', '.org', '.com']
country_list = ['.ca', '.fr', '.eu',]
image_format = ['JPG', 'TIFF', 'RAW', 'BMP', 'JPEG', 'PNG',
                'GIF', 'WEBP', 'PSD', 'HEIF', 'INDD',]  # scrap website page

# Need a secret key because the form requires it to be used in the html
app.config['SECRET_KEY'] = 'secretkey'
# defining where the uploaded picture are stored
app.config['PICTURES'] = PICTURES
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_FOLDER_SCRAP'] = UPLOAD_FOLDER_SCRAP
app.config['CROPPED_FOLDER'] = CROPPED_FOLDER
app.config['CROPPED_FOLDER_SCRAP'] = CROPPED_FOLDER_SCRAP
app.config['MODELS'] = MODELS

# ----------------------- CLASSES --------------------- #
# To Create a flask form that inherit from FlaskForm


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")


class URLForm(FlaskForm):
    url = StringField('URL', validators=[DataRequired(), URL()])
    submit = SubmitField('Submit')


# ---------------------- FUNCTIONS -------------------- #
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def img_preprocessing(image_path):

    image = utils.load_img(image_path,
                           grayscale=True,
                           color_mode="grayscale",
                           target_size=IMG_SIZE[:2],
                           interpolation="nearest",
                           keep_aspect_ratio=False,)

    input_arr = utils.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    return input_arr


def find_base_url(url):

    extensions_list = TLD_list + country_list  # Allowed list of URL extensions
    for tld in extensions_list:
        idx = url.find(tld)
        if idx != -1:  # If extension IS found in the URL, the base url is taken from the start 'http..' to the extension
            url_top = url[:(idx)]
            base_url = url_top + tld
    return base_url


def dectect_face(image_path):

    # datFile =  'static/library/' + 'shape_predictor_68_face_landmarks.dat'

    detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor(datFile)

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(image_path)
    # image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)

    return len(rects)


def cropface(image):
    detector = dlib.get_frontal_face_detector()
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(image)
    im = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # show the face number
        # cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
        # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # show the output image with the face detections
        face = image[y:y+h, x:x+w]
        # change from nparray to .jpg, etc ... image format
        face = image[max(0, y):min(y+h, image.shape[0]),
                     max(0, x):min(x+w, image.shape[1])]
        im = Image.fromarray(face)

        print('IMAGE WAS CROPPED')
        return im


def applycropface(inputfolder, outputfolder):
    count_im = 0
    num_images = 0
    for file in os.scandir(inputfolder):
        count_im += 1
        print('COUNT IMG:', count_im)
        print(f'current File: {file}')
        if file.is_file():  # and file.name[-4:]=='.jpg':
            file_path = f"{file.path}"
            print('IMAGE PATH:', file_path)
            with open(file_path, "r") as stream:
                print('stream:', stream.name)
                cropimage = cropface(stream.name)
                if cropimage is not None:
                    # os.chdir(CROPPED_FOLDER) # change the current working directory
                    # print("CWD:", os.getcwd())
                    # specify the name of the file to be saved
                    save_path = os.path.join(outputfolder, file.name)
                    print('SAVE TO:', save_path)
                    cropimage.save(save_path)
                    num_images += 1
                    print('CROPPED IMAGES: ', num_images)

                stream.close()


def predictions(image_path):
    print('Processing current img: ', image_path)
    # try:
    print('pre-processing')
    img = img_preprocessing(image_path)
    # USE WITH .pickle:
    # with open(MODELS + '/' + 'model_21_pckl.pkl','rb') as f:
    # USE WITH .h5:
    # model = tf.keras.models.load_model(MODELS + '/' + 'model_13_weights.h5')
    model = tf.keras.models.load_model(MODELS + '/' + 'model_21')

    print('Loading model')
    # model = pickle.load(f)
    print('Predicting')
    pred = model.predict(img)
    print('Prediction:', pred)
    predicted_class_indices = np.argmax(pred, axis=1)
    for k, v in labeldic.items():
        if v == predicted_class_indices:
            return k
    # except Exception:
    #     pass


# ------------------------ HOME Routes ----------------------- #

@app.route('/', methods=['GET', "POST"])
@app.route('/home', methods=['GET', "POST"])
def index():

    form_1 = UploadFileForm()
    print('f1:', form_1.file)
    if form_1.validate_on_submit():

        file = form_1.file.data  # get the file
        print('file:', file)
        # SAVING FILE
        if file:
            file.save(os.path.join(
                app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))  # save it Extra parameter: os.path.abspath(os.path.dirname(__file__)),

            print("The File has been uploaded")

    pics = os.listdir(UPLOAD_FOLDER)
    if pics:
        pics = [UPLOAD_FOLDER + f for f in pics]
        # THE FILE IS KEPT ONLY IF 1 FACE IS DETECTED
        for pic in (pics):
            try:
                num_faces = dectect_face(pic)
                if num_faces == 0 or num_faces > 1:
                    os.remove(pic)
                    pics.remove(pic)
            except Exception:
                # corrupted file, remove anyway
                os.remove(pic)
                pics.remove(pic)

        # CROP BEFORE PREDICTIONS
        applycropface(UPLOAD_FOLDER, CROPPED_FOLDER)
        croppics = os.listdir(CROPPED_FOLDER)
        croppics_path = [CROPPED_FOLDER + f for f in croppics]
        print("LIST OF CROPPED IMG TO PREDICT: ", croppics_path)

        # PREDICTIONS
        predList = [predictions(pic) for pic in croppics_path]

        return render_template('frontPage.html', form=form_1, pic_pred=zip(pics, predList))

    return render_template('frontPage.html', form=form_1)


# ------------------------ DELETE ----------------------------- #

# -------- ACCESS FROM INDEX --------- #
@app.route('/deleteAll_', methods=['GET', 'POST'])
def deleteall_():
    pics = os.listdir(UPLOAD_FOLDER)
    picscrop = os.listdir(CROPPED_FOLDER)
    if pics:
        for p in pics:
            os.remove(os.path.join(UPLOAD_FOLDER, p))
            if p in picscrop:
                os.remove(os.path.join(CROPPED_FOLDER, p))

    return redirect(url_for('index'))


@app.route('/delete/<string:get_img>/', methods=['GET'])
def deleteimg1(get_img):
    print('get_img:', get_img)
    pics = os.listdir(UPLOAD_FOLDER)
    picscrop = os.listdir(CROPPED_FOLDER)
    if get_img in pics:
        os.remove(os.path.join(UPLOAD_FOLDER, get_img))
    if get_img in picscrop:
        os.remove(os.path.join(CROPPED_FOLDER, get_img))
    return redirect(url_for('index'))


# ---------- ACCESS FROM SCRAP ------- #
@app.route('/deleteAll', methods=['GET', 'POST'])
def deleteall():
    pics = os.listdir(UPLOAD_FOLDER_SCRAP)
    picscrop = os.listdir(CROPPED_FOLDER_SCRAP)
    if pics:
        for p in pics:
            os.remove(os.path.join(UPLOAD_FOLDER_SCRAP, p))
            if p in picscrop:
                os.remove(os.path.join(CROPPED_FOLDER_SCRAP, p))

    return redirect(url_for('scrap'))


@app.route('/deleteimg/<string:get_img>/', methods=['GET'])
def deleteimg2(get_img):
    print('get_img:', get_img)
    pics = os.listdir(UPLOAD_FOLDER)
    picscrop = os.listdir(CROPPED_FOLDER)
    if get_img in pics:
        os.remove(os.path.join(UPLOAD_FOLDER_SCRAP, get_img))
    if get_img in picscrop:
        os.remove(os.path.join(CROPPED_FOLDER_SCRAP, get_img))

    return redirect(url_for('scrap'))


# ------------------------ SCRAP Route ------------------------- #
@app.route('/scrap', methods=['GET', "POST"])
def scrap():

    deleteall()

    form_2 = URLForm()
    print('f2:', form_2.url)

    if form_2.validate_on_submit():

        print("correct address")
        # SCRAPPING
        url = form_2.url.data
        page = requests.get(url)
        soup = bs(page.content, 'html.parser')
        images_tags_list = soup.find_all("img")
        try:
            images_uri_list = [tag.attrs['src'] for tag in images_tags_list]
        except Exception:
            return ' Sorry... we can not process this website'

        print('images_tags_list:', images_tags_list, '\n')
        print('images_uri_list:', images_uri_list)

        url_base = find_base_url(url)

        # Image URL. Either Full URL or website relative URL
        for i, img_uri in enumerate(images_uri_list):
            # If NOT a full URL , we rebuild the complete image URL adding the base URL
            if img_uri.find('http') == -1:
                full_img_url = url_base+'/'+img_uri
                images_uri_list[i] = full_img_url

        table = []
        pic_counter = 0
        for img_url in images_uri_list:  # Browse the image URL
            print('Browsing image URLs..')

            # Get request on full_url
            r = requests.get(img_url, stream=True)
            # 200 status code = OK | we process 30 pictures max
            if r.status_code == 200 and pic_counter <= ALLOWED_SCRAPPED_IMAGES:
                print('Image URL status code 200')
                # PREPARING IMAGE RECORDING TO THE LOCAL FOLDER
                # Retrieving the image NAME from the URL
                name = img_url.split('/')[-1]
                filepath = os.path.join(UPLOAD_FOLDER_SCRAP, name)
                print('filepath:', filepath)

                ext = os.path.splitext(name)[1].upper()

                if ext[1:] in image_format:  # check the IMAGE allowed formats
                    with open(filepath, 'wb') as f:  # RECORD IMAGE
                        r.raw.decode_content = True
                        # only record a picture if faces appear on it
                        shutil.copyfileobj(r.raw, f)
                        # THE PICTURE IS ONLY KEPT IF THERE IS 1 FACE ONLY
                        try:
                            num_detected_faces = dectect_face(filepath)
                            if num_detected_faces > 0 and num_detected_faces <= 1:  # This version only predicts for 1 face/picture
                                # add picture only if a face is detected
                                print('picture recorded for analysis')
                                pic_counter += 1
                                # prepare df inputs
                                # DELETE BUTTON: , filepath.split('/')[-1]]) #function to detect face => emotion
                                table.append([name, img_url, None])
                            else:
                                os.remove(filepath)
                        except Exception:
                            # corrupted file, remove anyway
                            os.remove(filepath)

        # CROP BEFORE PREDICTIONS

        print('table : ', table)

        applycropface(UPLOAD_FOLDER_SCRAP, CROPPED_FOLDER_SCRAP)
        # add the predictions to the table:
        croppics = os.listdir(CROPPED_FOLDER_SCRAP)
        croppics_path = [CROPPED_FOLDER_SCRAP + f for f in croppics]

        # zip_crop_pic_path = zip(croppics,croppics_path)
        print("LIST OF CROPPED IMG TO PREDICT: ", croppics_path)
        list_names = [(table_row, row[0])
                      for table_row, row in enumerate(table)]

        print('list_names:', list_names)

        # table iteration
        for table_row, name in list_names:
            print('table_row, name : ', table_row, name)
            zip_crop_pic_path = zip(croppics, croppics_path)
            for crpc_pth in zip_crop_pic_path:
                print('crpc_pth :', crpc_pth)
                if crpc_pth[0] == name:
                    print('table[table_row][2]', table[table_row][2])
                    print('prediction for :', crpc_pth[1])
                    table[table_row][2] = predictions(crpc_pth[1])
                    break

        # If no image could be processed
        if len(table) == 0:
            table.append(
                ['sorry these images can not be process', ' ', 'No prediction'])
            # Tranform df to html
            # DELETE BUTTON: 'Delete'])
            df = pd.DataFrame(table, columns=['Title', 'Image', 'Emotion',])
            return render_template('scrapPage.html', form=form_2, row_data=list(df.values.tolist()), titles=df.columns.values)

        # Tranform df to html
        # DELETE BUTTON: 'Delete'])
        df = pd.DataFrame(table, columns=['Title', 'Image', 'Emotion',])
        return render_template('scrapPage.html', form=form_2, row_data=list(df.values.tolist()), titles=df.columns.values)

    return render_template('scrapPage.html', form=form_2)


# ------------------------ ABOUT Route ----------------------- #

@app.route('/about', methods=['GET', "POST"])
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=False)
