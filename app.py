import os
import subprocess
import imageio
import ffmpeg
from flask import Flask, render_template, request, send_from_directory

import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model

app = Flask(__name__)

DATA_DIR = './data/s1'
OUTPUT_DIR = './static'

model = load_model()  # Load once globally

def convert_mpg_to_mp4(input_file, output_file):
    (
        ffmpeg
        .input(input_file)
        .output(output_file, vcodec='libx264')
        .run()
    )

@app.route('/', methods=['GET', 'POST'])
def index():
    options = os.listdir(DATA_DIR)
    selected_video = options[0] if options else None
    decoded_text = ""
    converted_prediction = ""
    
    if request.method == 'POST':
        selected_video = request.form.get('video')
        if selected_video:
            file_path = os.path.join(DATA_DIR, selected_video)
            output_path = os.path.join(OUTPUT_DIR, 'test_video.mp4')
            
            # Convert video to mp4
            convert_mpg_to_mp4(file_path, output_path)
            
            # Prepare model input
            video_data, annotations = load_data(tf.convert_to_tensor(file_path))
            
            # Create GIF for visualization
            gif_path = os.path.join(OUTPUT_DIR, 'animation.gif')
            imageio.mimsave(gif_path, video_data, duration=0.1)
            
            # Model prediction
            yhat = model.predict(tf.expand_dims(video_data, axis=0))
            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
            decoded_text = decoder
            converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
    
    return render_template('index.html', options=options, selected_video=selected_video,
                           decoded_text=decoded_text, converted_prediction=converted_prediction)

# Route to serve static files
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
