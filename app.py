import keras
from PIL import Image, ImageOps
import numpy as np
import streamlit as st

from classification import teachable_machine_classification
st.title("TB Classification with VGG")
st.text("Upload a CXR Image for Image Classification as TB or HEALTHY")

uploaded_file = st.file_uploader("Choose a CXR Image ...", type="png")
if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded CXR.', use_column_width=True)
        st.write("")
        # st.write("Classifying...")
        label = teachable_machine_classification(image)
        l=float(label[0][0])

        st.write(f"Normal {'-'*int(l*20)}|{'-'*int((1-l)*20)} TB")