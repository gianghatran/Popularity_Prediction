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


"""
# InstaPop Demo Time!

"""

img_df = pd.DataFrame(columns = ('Category_beauty',
                                'Category_family',
                                'Category_fashion',
                                'Category_fitness',
                                'Category_food',
                                'Category_interior',
                                'Category_other',
                                'Category_pet',
                                'Category_travel',
                                'weekday',
                                'aesthetic',
                                'tagged_acc',
                                'sponsored'))

#CREATE MENU FRAME FOR PREDICTION

#II. INPUT INFLUENCER CATEGORY

st.write('Ingredient 1: Influencer Category')

category = st.selectbox("Choose your category", ['beauty', 'family', 'food','fashion','travel','interior','fitness','pet','other'])
if category == 'beauty':
    img_df.loc[0,'Category_beauty'] = 1
    img_df.loc[0,'Category_family'] = 0
    img_df.loc[0,'Category_fashion'] = 0
    img_df.loc[0,'Category_fitness'] = 0
    img_df.loc[0,'Category_food'] = 0
    img_df.loc[0,'Category_interior'] = 0
    img_df.loc[0,'Category_other'] = 0
    img_df.loc[0,'Category_pet'] = 0
    img_df.loc[0,'Category_travel'] = 0

if category == 'family':
    img_df.loc[0,'Category_beauty'] = 0
    img_df.loc[0,'Category_family'] = 1
    img_df.loc[0,'Category_fashion'] = 0
    img_df.loc[0,'Category_fitness'] = 0
    img_df.loc[0,'Category_food'] = 0
    img_df.loc[0,'Category_interior'] = 0
    img_df.loc[0,'Category_other'] = 0
    img_df.loc[0,'Category_pet'] = 0
    img_df.loc[0,'Category_travel'] = 0

if category == 'fashion':
    img_df.loc[0,'Category_beauty'] = 0
    img_df.loc[0,'Category_family'] = 0
    img_df.loc[0,'Category_fashion'] = 1
    img_df.loc[0,'Category_fitness'] = 0
    img_df.loc[0,'Category_food'] = 0
    img_df.loc[0,'Category_interior'] = 0
    img_df.loc[0,'Category_other'] = 0
    img_df.loc[0,'Category_pet'] = 0
    img_df.loc[0,'Category_travel'] = 0

if category == 'fitness':
    img_df.loc[0,'Category_beauty'] = 0
    img_df.loc[0,'Category_family'] = 0
    img_df.loc[0,'Category_fashion'] = 0
    img_df.loc[0,'Category_fitness'] = 1
    img_df.loc[0,'Category_food'] = 0
    img_df.loc[0,'Category_interior'] = 0
    img_df.loc[0,'Category_other'] = 0
    img_df.loc[0,'Category_pet'] = 0
    img_df.loc[0,'Category_travel'] = 0

if category == 'food':
    img_df.loc[0,'Category_beauty'] = 0
    img_df.loc[0,'Category_family'] = 0
    img_df.loc[0,'Category_fashion'] = 0
    img_df.loc[0,'Category_fitness'] = 0
    img_df.loc[0,'Category_food'] = 1
    img_df.loc[0,'Category_interior'] = 0
    img_df.loc[0,'Category_other'] = 0
    img_df.loc[0,'Category_pet'] = 0
    img_df.loc[0,'Category_travel'] = 0

if category == 'interior':
    img_df.loc[0,'Category_beauty'] = 0
    img_df.loc[0,'Category_family'] = 0
    img_df.loc[0,'Category_fashion'] = 0
    img_df.loc[0,'Category_fitness'] = 0
    img_df.loc[0,'Category_food'] = 0
    img_df.loc[0,'Category_interior'] = 1
    img_df.loc[0,'Category_other'] = 0
    img_df.loc[0,'Category_pet'] = 0
    img_df.loc[0,'Category_travel'] = 0

if category == 'other':
    img_df.loc[0,'Category_beauty'] = 0
    img_df.loc[0,'Category_family'] = 0
    img_df.loc[0,'Category_fashion'] = 0
    img_df.loc[0,'Category_fitness'] = 0
    img_df.loc[0,'Category_food'] = 0
    img_df.loc[0,'Category_interior'] = 0
    img_df.loc[0,'Category_other'] = 1
    img_df.loc[0,'Category_pet'] = 0
    img_df.loc[0,'Category_travel'] = 0

if category == 'pet':
    img_df.loc[0,'Category_beauty'] = 0
    img_df.loc[0,'Category_family'] = 0
    img_df.loc[0,'Category_fashion'] = 0
    img_df.loc[0,'Category_fitness'] = 0
    img_df.loc[0,'Category_food'] = 0
    img_df.loc[0,'Category_interior'] = 0
    img_df.loc[0,'Category_other'] = 0
    img_df.loc[0,'Category_pet'] = 1
    img_df.loc[0,'Category_travel'] = 0

if category == 'travel':
    img_df.loc[0,'Category_beauty'] = 0
    img_df.loc[0,'Category_family'] = 0
    img_df.loc[0,'Category_fashion'] = 0
    img_df.loc[0,'Category_fitness'] = 0
    img_df.loc[0,'Category_food'] = 0
    img_df.loc[0,'Category_interior'] = 0
    img_df.loc[0,'Category_other'] = 0
    img_df.loc[0,'Category_pet'] = 0
    img_df.loc[0,'Category_travel'] = 1

#III. INPUT EXPECTED WEEKDAY OF POSTING
st.write('Ingredient 2: Posting Date')

weekday = st.selectbox("Choose the weekday you want to post", ['Monday', 'Tuesday', 'Wednesday','Thursday','Friday','Saturday','Sunday'])
if weekday == 'Monday':
    weekday = 0
    img_df.loc[0,'weekday'] = weekday

elif weekday == 'Tuesday':
    weekday = 1
    img_df.loc[0,'weekday'] = weekday

elif weekday == 'Wednesday':
    weekday = 2
    img_df.loc[0,'weekday'] = weekday

elif weekday == 'Thursday':
    weekday = 3
    img_df.loc[0,'weekday'] = weekday

elif weekday == 'Friday':
    weekday = 4
    img_df.loc[0,'weekday'] = weekday

elif weekday == 'Saturday':
    weekday = 5
    img_df.loc[0,'weekday'] = weekday

elif weekday == 'Sunday':
    weekday = 6
    img_df.loc[0,'weekday'] = weekday

else: 
    weekday == 'Input your weekday'

img_df.loc[0,'weekday'] = weekday


#V. TAGGED POST OR NOT
"""
Ingredient 3: Tagged people 
"""
tagged_post = st.selectbox("Will you tag anyone in this post?", ['yes', 'no'])

if tagged_post != None:
    '''
    Look at you all popular! Just one more step to greatness...
    '''
if tagged_post =="yes":
    tagged_post = 1

else: 
    tagged_post = 0

img_df.loc[0,'tagged_acc'] = tagged_post
'''
Ingredient 4: Sponsorship
'''
sponsorship = st.selectbox("Is this a sponsored post?", ['yes', 'no'])
if sponsorship == "yes":
    sponsorship = 1

else: 
    sponsorship = 0

img_df.loc[0,'sponsored'] = sponsorship

#V. Aesthetic Prediction

st.write('Ingredient 5: Image Aesthetic')
img_file = st.file_uploader("Choose a file")
if img_file is not None:

    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224,224))
    img_file.read()


    if aesthetic_pred(img):
        '''
        Our super complex machine is running, and it shows your image is of ....
        '''
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


    
        img_df.loc[0,'aesthetic'] = quality_score


#MODEL TRAINING
#prediction_2 = model2.predict(X)
prediction_2 = popular_prediction(img_df)

if prediction_2[0] == 0:
    prediction_2 = 'Not Popular'
elif prediction_2[0] == 1: 
    prediction_2 = 'Popular'

st.title('Your Post Will Be...')

if prediction_2 == 'Not Popular':
    st.write(prediction_2)
    st.image('https://pbs.twimg.com/profile_images/1042019157972320256/STolLU9B_400x400.jpg')
    '''
    Try a higher-quality image that is not pixelated, out of focused or blurry. This could help your post reach more people
    '''

if prediction_2 == 'Popular':
    st.write(prediction_2)
    st.image('https://winkgo.com/wp-content/uploads/2019/03/happy-memes-make-you-smile-more-31-720x680.jpg')
    '''
    Great job! Your post will be more likely to reach more audience and have better engagement. Keep up with the good work!
    '''