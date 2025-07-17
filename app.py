from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Pasta para salvar as imagens enviadas
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Carrega o modelo
model = load_model("modelo_carros.h5")

def preprocess_image(img_path, img_size=(255, 255)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            # Preprocessa e prediz
            img = preprocess_image(filepath)
            pred = model.predict(img)[0][0]
            label = "real" if pred >= 0.5 else "fake"
            return render_template('index.html', filename=file.filename, label=label, prob=pred)
    return render_template('index.html', filename=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)
