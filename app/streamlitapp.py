import streamlit as st
import os
import subprocess
import tensorflow as tf
print(os.getcwd())
from utils import load_data, num_to_char
from modelutil import load_model
import imageio

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="LipBuddy AI",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🗣️"
)

# -----------------------------
# Base directories
# -----------------------------
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data", "s1")
ALIGN_DIR = os.path.join(BASE_DIR, "data", "alignments", "s1")
OUTPUT_DIR = os.path.join(BASE_DIR, "app")
FFMPEG_PATH = os.path.join(BASE_DIR, "ffmpeg.exe")

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Controls")
    st.markdown("""
        **How to use:**  
        1. Select a video.  
        2. Click **Convert & Predict**.  
        3. View results in the main area.
    """)
    selected_video = st.selectbox("Select video", os.listdir(DATA_DIR))
    convert_predict = st.button("Convert & Predict")
    progress = st.progress(0)

# -----------------------------
# Main title
# -----------------------------
st.markdown("<h1 style='color:#FFF'>LipBuddy AI: Lip Reading Model</h1>", unsafe_allow_html=True)

# -----------------------------
# Main layout: two columns
# -----------------------------
col1, col2 = st.columns([1, 1])

# Placeholders
video_heading_placeholder = col1.empty()
video_placeholder = col1.empty()

gif_heading_placeholder = col2.empty()
gif_placeholder = col2.empty()

tokens_heading_placeholder = col2.empty()
tokens_placeholder = col2.empty()

decoded_heading_placeholder = col2.empty()
decoded_placeholder = col2.empty()

# Run conversion & prediction
if convert_predict and selected_video:
    try:
        video_path = os.path.join(DATA_DIR, selected_video)
        mp4_output = os.path.join(OUTPUT_DIR, "test_video.mp4")
        gif_output = os.path.join(OUTPUT_DIR, "animation.gif")

        # ---------- Step 1: Video conversion ----------
        progress.progress(10)
        subprocess.run([
            FFMPEG_PATH,
            "-y",
            "-i", video_path,
            "-vcodec", "libx264",
            "-acodec", "aac",
            mp4_output
        ], check=True, shell=True)
        progress.progress(40)

        # Display original video

        video_heading_placeholder.markdown("<div style='font-size:20px; font-weight:bold; color:white;'>Original Video</div>", unsafe_allow_html=True)
        video_placeholder.video(mp4_output)

        # ---------- Step 2: Load video frames ----------
        video_frames, annotations = load_data(tf.convert_to_tensor(video_path))
        progress.progress(60)

        # ---------- Step 3: Create AI visualization GIF ----------
        imageio.mimsave(gif_output, video_frames, fps=10)
        gif_heading_placeholder.markdown("<div style='font-size:20px; font-weight:bold; color:white;'>AI Visualization</div>", unsafe_allow_html=True)
        gif_placeholder.image(gif_output, width=400)
        progress.progress(75)

        # ---------- Step 4: Model prediction ----------
        model = load_model()
        yhat = model.predict(tf.expand_dims(video_frames, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        decoded_text = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode("utf-8")
        progress.progress(90)

        # Display raw tokens on left column
        tokens_heading_placeholder.markdown("<div style='font-size:20px; font-weight:bold; color:white;'>Raw Model Tokens</div>", unsafe_allow_html=True)
        tokens_placeholder.code(decoder)

        # Display decoded text on right column
        decoded_heading_placeholder.markdown("<div style='font-size:20px; font-weight:bold; color:white;'>Decoded Text</div>", unsafe_allow_html=True)
        decoded_placeholder.markdown(f"<div style='font-size:22px; color:#FFF'>{decoded_text}</div>", unsafe_allow_html=True)

        progress.progress(100)

    except subprocess.CalledProcessError as e:
        st.error(f"Video conversion failed: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
