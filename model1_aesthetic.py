import tensorflow as tf
from tensorflow.keras import models,layers
from tensorflow import keras

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

import pathlib
import pandas as pd
from PIL import Image
from numpy import asarray
import os

from model2_ML import *

def aesthetic_pred(img):
    IMG_SIZE = 224  
    BATCH_SIZE = 32

    model = tf.keras.models.load_model('/Users/user/Desktop/Post_image/model_save/best_model',custom_objects={'imagenet_utils': imagenet_utils})
# convert image to numpy array
    #data = asarray(img.resize((224,224)))
    data = np.expand_dims(img,0)
    #data = data.reshape(1,224,224,3)
    prob_index = np.argmax(model.predict(data)[0], axis=0, out=None)

    if prob_index ==0: 
        aes_predict = "high_quality"

    if prob_index ==1:
        aes_predict = "medium_quality"

    if prob_index == 2: 
        aes_predict = 'low_quality'
    
    return aes_predict



