import streamlit as st 
import cv2
from PIL import Image
import numpy as np

# function to convert the image into grayscale
def convert_to_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

# define function to detect edges 
def detect_edges(img):
    edges=cv2.Canny(img, 100, 200)
    return edges 

# function to detect faces in the image
def detect_faces(img):
    face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 6)
    return img 

# set the title of the web app 
st.title('OpenCv App for small image processing task')

# add a button to upload the image file from user 
uploaded_file=st.file_uploader('Choose an image...', type=['jpg', 'png', 'jpeg'])
if uploaded_file is not None:
    # Convert the file to an opencv image. 
    file_bytes=np.asarray(bytearray(uploaded_file. read()), dtype=np.uint8)
    img=cv2.imdecode(file_bytes, 1)
    
    # Display the oroginal image
    st.image(img, channels='BGR', use_column_width=True)

    # When the 'Grayscale' button is clicked, convert the image to grayscale
    if st.button('Convert to Grayscale'):
        img_gray=convert_to_gray(img)
        st.image(img_gray, use_column_width=True)

    # When the 'Detect Edges' button is clicked, detect edges in the image 
    if st.button ('Detect Edges'):
        img_edges=detect_edges(img)
        st.image(img_edges, use_column_width=True)

    # When the 'Detect Faces' button is clicked, detect faces in the image
    if st.button ('Detect Faces'):
        img_faces=detect_faces(img)
        st.image(img_faces, channels='BGR', use_column_width=True)
        
