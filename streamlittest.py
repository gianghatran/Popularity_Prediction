import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time 
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

import pathlib
import pandas as pd
from PIL import Image
from numpy import asarray
import os

from model1_aesthetic import *
from model2_ML import *

menu = ['Aesthetic Prediction','Recommendation']
choice = st.sidebar.selectbox('Select one', menu)

if choice == 'Aesthetic Prediction':
    img_file = st.file_uploader("Choose a file")
    if img_file is not None:

        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 3)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(224,224))
        img_file.read()
        '''
        Our super complex machine is running, and it shows your image is of ....
        '''
        if aesthetic_pred(img):

            st.write(aesthetic_pred(img))


            if aesthetic_pred(img) == 'low_quality':
                quality_score = 0
                st.write ('Consider change your image for better engagement')
            elif aesthetic_pred(img) == 'medium_quality':
                quality_score = 1
                st.write ('Almost there! Improve image focus with definitely help!')

            elif aesthetic_pred(img) == 'high_quality': 
                quality_score = 2
                st.write ('Amazing! Some pretty thing you have here!')