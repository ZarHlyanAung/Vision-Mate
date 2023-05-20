import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Define the class labels
class_labels = ["1000", "10000", "500", "5000", "other"]

def loadModel():
    model = tf.keras.models.load_model("vision-mate-4.h5")
    return model


model = loadModel()

# Define the Streamlit app
st.title('Vision Mate Demo')
st.sidebar.title('Upload Image')
    
# Create a file uploader in the sidebar
uploaded_file = st.sidebar.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
    
if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
        
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
        
    # Preprocess the image for model input
    image = image.resize((100, 100)) # Resize the image to match the input size of the model
    image = np.array(image) # Convert the image to a numpy array
    image = image / 255.0 # Normalize the pixel values to [0, 1]
    image = np.expand_dims(image, axis=0) # Add an extra dimension for batch size
        
    # Make predictions with the model
    prediction = model.predict(image)
    predicted_label_index = np.argmax(prediction[0])
    predicted_label = class_labels[predicted_label_index]
        
    # Display the predicted label
    st.success(predicted_label)
    st.write(prediction[0][predicted_label_index])
