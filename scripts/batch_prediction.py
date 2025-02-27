import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("sugarcane_disease_model.h5")

# Define class names
class_names = ["healthy", "RedRot", "RedRust"]

# Disease information
disease_info = {
    "healthy": {
        "name": "Healthy",
        "solution": "The plant is healthy. No action required."
    },
    "RedRot": {
        "name": "Red Rot",
        "solution": "1. Remove and destroy infected plants. 2. Apply fungicides like Carbendazim. 3. Practice crop rotation."
    },
    "RedRust": {
        "name": "Red Rust",
        "solution": "1. Remove infected leaves. 2. Apply fungicides like Mancozeb. 3. Ensure proper spacing between plants."
    }
}

# Function to predict disease
def predict_disease(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(128, 128))  # Match the input size of your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

    # Predict the disease
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    class_name = class_names[predicted_class]

    # Get disease information
    disease_name = disease_info[class_name]["name"]
    solution = disease_info[class_name]["solution"]

    return disease_name, solution

# Function to process all images in a folder
def process_folder(folder_path):
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Process each image
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        disease_name, solution = predict_disease(image_path)

        # Display the results
        print(f"Image: {image_file}")
        print(f"Disease: {disease_name}")
        print(f"Solution: {solution}")
        print("-" * 50)

# Take user input for the folder path
folder_path = input("Enter the path to the folder containing images: ")

# Process all images in the folder
process_folder(folder_path)
