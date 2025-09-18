import os
import subprocess
import imageio
import ffmpeg
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from utils import load_data, num_to_char
from modelutil import load_model

app = FastAPI(title="LipReader API")

# Load model once on startup
model = load_model()

DATA_DIR = os.path.join(".", "data", "s1")
OUTPUT_VIDEO = "./app/test_video.mp4"
ANIMATION_GIF = "./app/animation.gif"


def convert_mpg_to_mp4(input_file, output_file):
    (
        ffmpeg
        .input(input_file)
        .output(output_file, vcodec="libx264")
        .run(overwrite_output=True)
    )


@app.get("/videos")
def list_videos():
    """List available video files"""
    options = os.listdir(DATA_DIR)
    return {"videos": options}


@app.get("/video/{filename}")
def get_video(filename: str):
    """Convert and return video in mp4 format"""
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse({"error": "File not found"}, status_code=404)

    convert_mpg_to_mp4(file_path, OUTPUT_VIDEO)
    return FileResponse(OUTPUT_VIDEO, media_type="video/mp4")


@app.get("/predict/{filename}")
def predict(filename: str):
    """Run model prediction on video"""
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse({"error": "File not found"}, status_code=404)

    # Load data as frames
    video, annotations = load_data(tf.convert_to_tensor(file_path))
    imageio.mimsave(ANIMATION_GIF, video, duration=0.1)

    # Model prediction
    yhat = model.predict(tf.expand_dims(video, axis=0))
    decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()

    # Convert to text
    converted_prediction = (
        tf.strings.reduce_join(num_to_char(decoder)).numpy().decode("utf-8")
    )

    return {
        "tokens": decoder.tolist(),
        "prediction": converted_prediction,
        "animation": "/animation",
    }


@app.get("/animation")
def get_animation():
    """Return gif animation of model's perspective"""
    if not os.path.exists(ANIMATION_GIF):
        return JSONResponse({"error": "No animation available"}, status_code=404)
    return FileResponse(ANIMATION_GIF, media_type="image/gif")
