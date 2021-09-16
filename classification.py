import keras
from keras.models import model_from_json
from PIL import Image, ImageOps
import numpy as np
import cv2
from data import *
import streamlit as st


def teachable_machine_classification(img):
    # Load the model
    json_file = open('models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("models/tb_detector_weights.best.hdf5")

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    size = (224, 224)
    img = ImageOps.fit(img, size, )
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = np.asarray(img)
    data[0]= img

    # run the inference
    prediction = model.predict(data)
    return prediction  # return position of the highest probability


