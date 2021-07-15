import streamlit as st
from PIL import Image
import numpy as np
import cv2
import requests
from io import BytesIO
from deepface import DeepFace

st.title("Face Verification using DeepFace")

def face_verification(col, image):
  col.image(image, caption="Selected Image", use_column_width=True)
  obj = DeepFace.analyze(image, actions = ['age', 'gender', 'race', 'emotion'])
  col.write("Age: "+str(obj['age']))
  col.write("Gender: "+ str(obj['gender']))
  col.write("Race: "+ str(obj['dominant_race']))
  col.write("Emotion: "+ str(obj['dominant_emotion']))


col1, col2 = st.beta_columns(2)
img_file_buffer1 = col1.file_uploader("Upload image or select sample images from box", type=["png", "jpg", "jpeg"], key="im1")
img_file_buffer2 = col2.file_uploader("Upload image or select sample images from box", type=["png", "jpg", "jpeg"], key="im2")

if img_file_buffer1 is not None:
  image1 = np.array(Image.open(img_file_buffer1))
  face_verification(col1, image1)

if img_file_buffer2 is not None:
  image2 = np.array(Image.open(img_file_buffer2))
  face_verification(col2, image2)

if (img_file_buffer1 is not None) and (img_file_buffer2 is not None):
  st.title("Verify faces")
  metrics = ["cosine", "euclidean", "euclidean_l2"]
  result = DeepFace.verify(image1, image2, distance_metric=metrics[2])
  
  if(result['verified']):
    st.write("Same person")
    st.write(result)
  else:
    st.write("Not the same person")
    st.write(result)