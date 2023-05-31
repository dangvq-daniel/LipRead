# Import all of the dependencies

import numpy as np
import streamlit as st
import os
import imageio
import ffmpeg
import subprocess

import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipReader')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('Lip Net Full Stack App')
st.text(os.listdir())

options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose Video', options)

# Generate two columns
col1, col2 = st.columns(2)

def convert_mpg_to_mp4(input_file, output_file):
    (
        ffmpeg
        .input(input_file)
        .output(output_file, vcodec='libx264')
        .run()
    )

if options:
    
    with col1:
        st.info('The video blow displays the converted video in mp4 format')
        file_path = os.path.join('..','data','s1', selected_video)
        output_path = os.path.join('..', 'data', 's1', 'test_video.mp4')
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 {output_path} -y')

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)

    with col2:
        st.info('This is what the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        imageio.mimsave('animation.gif', video, duration = 0.1)
        st.image('animation.gif', width = 400)

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
