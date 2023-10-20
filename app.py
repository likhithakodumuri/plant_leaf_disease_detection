import os
from flask import Flask, render_template, request, jsonify
from utils.predict import predict_disease

app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = "E:/AU TPCELL/MINI_PROJECT/Final_Mini_Project/plant_leaf_disease_detection/static/uploaded images"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the uploaded image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        # Call the prediction function
        result = predict_disease(image_path)

        # Return the prediction result
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
