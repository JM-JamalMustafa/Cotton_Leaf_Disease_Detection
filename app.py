# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 22:34:20 2020

@author: Krish Naik
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'C:/Users/Jamal/Downloads/Compressed/2/model_inceptionv3.h5'

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)

    # Make prediction
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)

    # Map class indices to labels
    class_labels = {0: 'The leaf is diseased cotton leaf',
                    1: 'The leaf is diseased cotton plant',
                    2: 'The leaf is fresh cotton leaf',
                    3: 'The leaf is fresh cotton plant'}
    preds_label = [class_labels[pred] for pred in preds]

    return preds_label[0]


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


import os

# ...

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
                                                                                                                                                                                                                                                                                            
        # Create the uploads directory if it does not exist
        basepath = os.path.dirname(__file__)
        uploads_path = os.path.join(basepath, 'uploads')
        if not os.path.exists(uploads_path):
            os.makedirs(uploads_path)

        # Save the file to ./uploads
        file_path = os.path.join(uploads_path, secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result = preds

        # Display the prediction result
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001,debug=True)
