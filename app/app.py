# Import all of the dependencies
import streamlit as st
import os 
import imageio 
import subprocess
import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('LipNet Full Stack App') 
# Generating a list of options or videos 
options = os.listdir(os.path.join('.', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('.','data','s1', selected_video)
        FFMPEG_PATH = os.path.join(os.getcwd(), "ffmpeg.exe")  # ./ffmpeg.exe
        INPUT_PATH = file_path  # your video input path
        OUTPUT_PATH = os.path.join(os.getcwd(), "app", "test_video.mp4")

        # Convert video to MP4
        subprocess.run([
            FFMPEG_PATH,
            "-y",                  # overwrite output
            "-i", INPUT_PATH,      # input file
            "-vcodec", "libx264",  # video codec
            "-acodec", "aac",       # audio codec
            OUTPUT_PATH
        ], check=True)
        # Rendering inside of the app
        video = open(os.path.join(os.getcwd(), "app", "test_video.mp4"), 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)


    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        imageio.mimsave(os.path.join(os.getcwd(), "app", "animation.gif"), video, fps=10)
        st.image(os.path.join(os.getcwd(), "app", "animation.gif"), width=400) 

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
        