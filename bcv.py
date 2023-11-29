import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Function to apply image processing techniques
def apply_image_processing(image, technique):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply the selected technique
    if technique == 'Canny':
        processed_image = cv2.Canny(gray_image, 50, 150)
    elif technique == 'Log':
        processed_image = cv2.Laplacian(gray_image, cv2.CV_64F)
        processed_image = np.uint8(np.absolute(processed_image))
    elif technique == 'Dog':
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        processed_image = cv2.filter2D(gray_image, -1, kernel)
    else:
        processed_image = gray_image

    return processed_image

# Streamlit app
def main():
    st.title("Image Analyzer App")

    # Upload image through Streamlit
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Get the image as a NumPy array
        image = np.array(Image.open(uploaded_image))

        # Image processing options
        techniques = ['Original', 'Canny', 'Log', 'Dog']
        selected_technique = st.selectbox("Select Image Processing Technique", techniques)

        # Apply image processing
        if selected_technique != 'Original':
            processed_image = apply_image_processing(image, selected_technique)
            st.image(processed_image, caption=f"{selected_technique} Processed Image", use_column_width=True)

# Run the Streamlit app
if __name__ == '__main__':
    main()
