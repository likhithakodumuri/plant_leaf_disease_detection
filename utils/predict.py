import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Define the class labels
class_labels = [
    "Tomato-Bacterial spot",
    "Tomato-Early blight",
    "Tomato-Healthy",
    "Tomato-Late blight",
    "Tomato-Leaf Mold",
    "Tomato-Septoria_leaf_spot",
    "Tomato-Target Spot",
    "Tomato-Tomato_mosaic_virus",
    "Tomato-Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato-Two-spotted spider mite"
]

# Load the pre-trained CNN model
model = load_model("E:/AU TPCELL/MINI_PROJECT/Final_Mini_Project/plant_leaf_disease_detection/models/Leaf_Classification.h5")

# Define a function to make predictions
def predict_disease(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Make predictions
    predictions = model.predict(img)

    # Get the predicted class and confidence
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]
    confidence = predictions[0][predicted_class_index]

    # Create the result dictionary
    result = {
        'predicted_class': predicted_class,
        'confidence': float(confidence)
    }

    return result
