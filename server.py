from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
import torch
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load custom YOLO model
model = YOLO('best.pt')  # Replace 'best.pt' with your model file

def run_inference(image_path):
    results = model(image_path)
    for result in results:
        img = result.plot()
        output_path = os.path.join(app.config['RESULT_FOLDER'], os.path.basename(image_path))
        cv2.imwrite(output_path, img)
        return output_path
    return None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        result_path = run_inference(filepath)
        return redirect(url_for('display_result', filename=os.path.basename(result_path)))
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def display_result(filename):
    return render_template('result.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
