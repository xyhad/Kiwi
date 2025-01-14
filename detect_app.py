from tensorflow.keras.applications.inception_v3 import InceptionV3
#import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

def import_and_predict(image_data, model):
    
    size = (75,75) #set the image size
    image = ImageOps.fit(image_data, size) #prepare image with anti-aliasing
    image = image.convert('RGB') #convert image to RGB, Red Green Blue format
    image = np.asarray(image) #convert image into array
    image = (image.astype(np.float32) / 255.0) #create image array matrix
    img_reshape = image[np.newaxis,...] #np.newaxis will create new dimension
    prediction = model.predict(img_reshape) #give predicted output based on the input image
    
    return prediction #return predicted output

model = tf.keras.models.load_model('C:/Python/rps/my_model.hdf5')
pre_trained_model = InceptionV3(input_shape = (75, 75, 3),
include_top = False,
weights = 'imagenet')

st.write("""
# Rock-Paper-Scissor Hand Sign Prediction
"""
)

st.write("This is a simple image classification web app to predict rock-paper-scissor hand sign")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])


if file is None:
    st.text("You haven't uploaded an image file")

else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)

    if np.argmax(prediction) == 0:
        st.write("It is a paper!")

    elif np.argmax(prediction) == 1:
        st.write("It is a rock!")

    else:
        st.write("It is a scissor!")

    st.text("Probability (0: Paper, 1: Rock, 2: Scissor)")
    st.write(prediction)