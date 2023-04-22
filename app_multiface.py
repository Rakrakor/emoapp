# Set up Flask: https://www.youtube.com/watch?v=GHvj1ivQ7ms
# Upload a picture: https://www.youtube.com/watch?v=GeiUTkSAJPs
# In VSCODE: CTRL + SHIFT + P => search for 'PYTHON INTERPRETER' => select env
# In Terminal : source env/bin/activate => activate the virtual environment


from flask import Flask, render_template, request, redirect, url_for, flash
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
import pickle

import numpy as np 
import pandas as pd
#NN libraries
import tensorflow as tf
#print(tf.__version__)
from tensorflow import keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.mobilenet_v2 import MobileNetV2

#image processing
from glob import glob
#from PIL import Image as PIL_Image
from keras import utils
from keras.preprocessing import image as keras_image
from keras.preprocessing.image import ImageDataGenerator

# FaceFeatures detection
import cv2 
import dlib
import imutils
from imutils import face_utils


app = Flask(__name__)

IMG_SIZE=[48,48,3]
UPLOAD_FOLDER = 'static/files'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'} #upload image page
labeldic = {'angry': 0, 'disgust': 1,'fear': 2,'happy': 3,'neutral': 4,'sad': 5,'surprise': 6}
TLD_list = ['.edu','.info','.biz','.co','.gov','.mil','.me', '.org','.com']
country_list =['.ca','.fr','.eu',]
image_format=['JPG', 'TIFF','RAW','BMP', 'JPEG', 'PNG','GIF', 'WEBP','PSD','HEIF','INDD',] #scrap website page

# Need a secret key because the form requires it to be used in the html
app.config['SECRET_KEY'] = 'secretkey'
# defining where the uploaded picture are stored
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create a flask form that inherit from FlaskForm


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")



def img_preprocessing(image_path,prepro=True):
    print('PRE PROC_1')
    print('image shape: ', image_path.shape)
    if prepro:
        print('PRE PROC_2')
        image = utils.load_img(image_path,grayscale=False,color_mode="rgb",target_size=IMG_SIZE[:2],interpolation="nearest",keep_aspect_ratio=False,)
    
    print('PRE PROC_3')
    input_arr = utils.img_to_array(image)
    print('PRE PROC_4')    
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    print('PRE PROC_5_FINAL')
    return input_arr


def detect_face(image_path, copy=True):
    print('DETECT_FACE_1')
    #datFile =  'static/library/' + 'shape_predictor_68_face_landmarks.dat'

    detector = dlib.get_frontal_face_detector()
    #predictor = dlib.shape_predictor(datFile)

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(image_path)
    # if copy==True:
    #     image = image.copy

    print('DETECT_FACE_2')
    image = imutils.resize(image, width=500)
    print('DETECT_FACE_3')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print('DETECT_FACE_4')
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    print('DETECT_FACE_5_FINAL')

    return rects, image
    

def predict_face(rect, image):
    # Extract the face from the image using the rectangle
    print('predict_face: STEP_1')
    print('rect: ', rect)
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    print('predict_face: STEP_2')
    face = image[y:y+h, x:x+w]
    print('predict_face: STEP_3')
    # Preprocess the face for the CNN model
    try:
        #img = img_preprocessing(face,prepro=True)
        img=face # otherwise use pre-processinf function
        with open('static/model/model_11_pckl.pkl','rb') as f:
            model = pickle.load(f)
            
            print('predict_face: STEP_4')            
            pred = model.predict(img)
            print('Prediction:', pred)
            print('predict_face: STEP_5')
            predicted_class_indices=np.argmax(pred,axis=1)
            print('predict_face: STEP_6')
            for emo_label,v in labeldic.items():
                if v==predicted_class_indices:
                    print('predict_face: STEP_7_FINAL')
                    return emo_label
            
    except Exception:
        print('predict_face EXCEPTION: STEP_8')
        pass


def tag_faces(image_path):
    print('tag_faces: STEP_1')
    rects, image = detect_face(image_path,False)
    print('tag_faces: STEP_2')
    # Loop over the face detections
    for (i, rect) in enumerate(rects):
        print('tag_faces: STEP_3')
        # Predict the class label for the face
        label = predict_face(rect, image)
        print('tag_faces: STEP_4')

        # Draw a rectangle around the face
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        print('tag_faces: STEP_5')
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print('tag_faces: STEP_6')
        # Display the predicted class label on the image
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print('tag_faces: STEP_7_FINAL')


    return image


    

#@app.route('/prediction', methods=['GET', "POST"])
def predictions(image_path):
    print('Processing current img: ',image_path)
    try:
        img = img_preprocessing(image_path)
        with open('static/model/model_11_pckl.pkl','rb') as f:
            model = pickle.load(f)
            
            pred = model.predict(img)
            print('Prediction:', pred)
            predicted_class_indices=np.argmax(pred,axis=1)
            for k,v in labeldic.items():
                if v==predicted_class_indices:
                    return k
    except Exception:
        pass



@app.route('/', methods=['GET', "POST"])
@app.route('/home', methods=['GET', "POST"])
def index():
    
    form_1 = UploadFileForm()
    print('f1:',form_1.file)
    if form_1.validate_on_submit():
        
        file = form_1.file.data  # get the file
        print('file:', file)
        if file: #and allowed_file(file.filename)
            
            file.save(os.path.join(
                    app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))  # save it Extra parameter: os.path.abspath(os.path.dirname(__file__)),
            
            print("The File has been uploaded")

        pics = os.listdir('static/files')
        if pics:
            pics = ['static/files/'+ f for f in pics]
            
            for pic in (pics):
                try:
                    num_faces = len(detect_face(pic)[0])
                    if num_faces==0:
                        os.remove(pic)
                        pics.remove(pic)
                except Exception:
                    #corrupted file, remove anyway
                    os.remove(pic)
                    pics.remove(pic)
                            

            predList = [predictions(pic) for pic in pics]
            
        return render_template('frontPage.html',form=form_1, pic_pred=zip(pics,predList))

    return render_template('frontPage.html', form=form_1)



# ------------------------ DELETE ----------------------------- #
@app.route('/deleteAll', methods=['GET','POST'])
def deleteall():
    pics = os.listdir('static/files')
    if pics:
        for p in pics:
            os.remove(os.path.join('static/files/',p))
        
    return redirect(url_for('scrap'))

@app.route('/deleteAll_', methods=['GET','POST'])
def deleteall_():
    pics = os.listdir('static/files')
    if pics:
        for p in pics:
            os.remove(os.path.join('static/files/',p))
        
    return redirect(url_for('index'))



@app.route('/delete/<string:get_img>/', methods=['GET'])
def deleteimg1(get_img):
    print('get_img:',get_img)
    
    os.remove(os.path.join(UPLOAD_FOLDER,get_img))
    return redirect(url_for('index'))


@app.route('/deleteimg/<string:get_img>/', methods=['GET'])
def deleteimg2(get_img):
    print('get_img:',get_img)
    
    os.remove(os.path.join(UPLOAD_FOLDER,get_img))

    # #UPDATE
    # global mem_df
    # mem_df.drop( mem_df[( mem_df['image']==get_img)].index, inplace=True)
    
    return redirect(url_for('scrap'))




# ------------------------ SCRAP Page------------------------- #


from sys import stderr
import requests
from bs4 import BeautifulSoup as bs

import shutil

#from IPython.display import Image, HTML
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, URL


def find_base_url(url):
    
    extensions_list = TLD_list + country_list
    for tld in extensions_list:
        idx=url.find(tld)
        if idx!=-1:
            url_top = url[:(idx)]
            base_url = url_top + tld
    return base_url



class URLForm(FlaskForm):
    url = StringField('URL', validators=[DataRequired(), URL()])
    submit = SubmitField('Submit')


#global mem_df
mem_df = pd.DataFrame
@app.route('/scrap', methods=['GET', "POST"])
def scrap():
    
    form_2 = URLForm()
    print('f2:',form_2.url)
    

    # #UPDATE
    # if mem_df:
    #     return render_template('scrapPage.html', form=form_2, row_data=list(df.values.tolist()),titles=df.columns.values)


    if form_2.validate_on_submit():
        print("correct address")
        url = form_2.url.data
        page = requests.get(url)
        soup = bs(page.content, 'html.parser')
        images_tags_list = soup.find_all("img")
        try:
            images_uri_list = [tag.attrs['src'] for tag in images_tags_list]  # add try / Except block
        

            print('images_tags_list:',images_tags_list,'\n')
            print('images_uri_list:',images_uri_list)
            

            url_base = find_base_url(url)

            for i, img_uri in enumerate(images_uri_list):
                if img_uri.find('http')==-1:
                    full_img_url = url_base+'/'+img_uri
                    images_uri_list[i] = full_img_url
            
            table =[]
            pic_counter =0
            for img_url in images_uri_list:
                
                r = requests.get(img_url, stream=True) 
                #Get request on full_url
                
                if r.status_code == 200 and pic_counter<=30:                     
                #200 status code = OK
                    name = img_url.split('/')[-1]
                    filepath = os.path.join(UPLOAD_FOLDER, name)
                    print('FILEPATH:', filepath)

                    ext = os.path.splitext(name)[1].upper()
                    #throws error load_img img = pil_image.open(io.BytesIO(f.read()))
                    if ext[1:] in image_format: # add check face condition
                        with open(filepath, 'wb') as f: 
                            r.raw.decode_content = True
                            #only record a picture if faces appear on it
                            shutil.copyfileobj(r.raw, f)
                            print('SCRAP - FILE CORRECTLY RECORDED')
                            try:
                                #num_faces = len(detect_face(filepath)[0])
                                num_faces = 1
                                print('SCRAP - num_faces: ', num_faces)
                                if num_faces>0:
                                    pic_counter+=1 
                                    print('SCRAP - THIS PICTURE HAS FACES')
                                    try:
                                        tagged_image = tag_faces(filepath)
                                        with open(filepath, 'wb') as f: 
                                            print('SCRAP - FILE OPENING')
                                            shutil.copyfileobj(tagged_image, f)
                                    except Exception as e:
                                        print(e)
                                else:
                                    os.remove(filepath)
                                    print('SCRAP - NO FACE')
                            except Exception:
                                #corrupted file, remove anyway
                                print('SCRAP - EXCEPTION FILE REMOVED - NO FACE DETECTED:', filepath)
                                os.remove(filepath)

                            # ----- Update
                            
                                        

                            # ------------

                            #prepare df to html
                            #add picture only if a face is detected
                            #table.append([name,img_url, predictions(filepath), filepath.split('/')[-1]]) #function to detect face => emotion
                            table.append([name,img_url, 'See Picture', filepath.split('/')[-1]]) #function to detect face => emotion
                        
                            
            
                        
            df = pd.DataFrame(table, columns=['Title', 'Image', 'Emotion','Delete'])

            # #UPDATE
            # global mem_df 
            # mem_df = df

            return render_template('scrapPage.html', form=form_2, row_data=list(df.values.tolist()),titles=df.columns.values)
        except Exception:
            return 'can not process this website'

        

    
    return render_template('scrapPage.html', form=form_2)
    
    




if __name__ == "__main__":
    app.run(debug=True)
