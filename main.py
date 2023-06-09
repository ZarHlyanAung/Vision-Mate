import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Define the class labels
class_labels = ["100", "1000", "10000", "200", "50", "500", "5000", "other"]


def loadModel():
    model = tf.keras.models.load_model("vision-mate-13.h5")
    return model


model = loadModel()

# Define the Streamlit app
st.title('Vision Mate Demo')

# Create a file uploader
uploaded_file = st.file_uploader(
    'Upload an image of a Myanmar bank note', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)

    # Convert image to grayscale
#     grayscale_image = image.convert('L')

    # Display the uploaded grayscale image
    st.image(image, caption='Grayscale Image', use_column_width=True)

    # Preprocess the image for model input
    # Resize the image to match the input size of the model
    resized_image = image.resize((150, 150))
    image_array = np.array(resized_image)  # Convert the image to a numpy array
    # Normalize the pixel values to [0, 1]
    normalized_image = image_array / 255.0
    # Add an extra dimension for batch size
    input_image = np.expand_dims(normalized_image, axis=0)

    # Make predictions with the model
    prediction = model.predict(input_image)
    predicted_label_index = np.argmax(prediction[0])
    predicted_label = class_labels[predicted_label_index]

    # Display the predicted label
    st.success(predicted_label)
    st.write(prediction[0][predicted_label_index])
