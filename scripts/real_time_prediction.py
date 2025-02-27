import cv2
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("sugarcane_disease_model.h5")

# Define class names
class_names = ["healthy", "RedRot", "RedRust"]

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Preprocess the frame
    resized_frame = cv2.resize(frame, (128, 128))  # Match the input size of your model
    img_array = tf.keras.preprocessing.image.img_to_array(resized_frame)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

    # Predict the disease
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    class_name = class_names[predicted_class]

    # Display the prediction on the frame
    cv2.putText(frame, f"Disease: {class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Sugarcane Disease Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
