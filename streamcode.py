import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
model = load_model('plant.h5')
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

st.set_page_config(page_title="Miniproject", page_icon=":camera:", layout="wide")
prediction="None"
st.title("Detection of Plant diseases in a CNN based approach") 

uploaded_file = st.file_uploader("Insert a plant image...", type=["jpg", "jpeg", "png"])
rescale = Rescaling(scale=1.0/255)
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.resize((256, 256))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    #image = rescale(image)
    #st.image(image, caption="Output :", use_column_width=True)
    prediction = model.predict(image)
    st.caption("Detected Disease Class is ")
    classes=["Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy",
                "Blueberry___healthy","Cherry_(including_sour)__Powedery_mildew","Cherry_(including_sour)__healthy",
                "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_","Corn_(maize)___Northern_Leaf_Blight",
                "Corn_(maize)___healthy","Grape___Black_rot","Grape___Esca_(Black_Measles)","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
                "Grape___healthy","Orange___Haunglongbing_(Citrus_greening)","Peach___Bacterial_spot","Peach___healthy",
                "Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy","Potato___Early_blight","Potato___Late_blight",
                "Potato___healthy","Raspberry___healthy","Soybean___healthy","Squash___Powdery_mildew",
                "Strawberry___Leaf_scorch","Strawberry___Healthy","Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___Late_blight",
                "Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot","Tomato___Spider_mites Two-spotted_spider_mite",
                "Tomato___Target_Spot","Tomato_Yellow_Leaf_Curl_Virus","Tomato_mosaic_virus","Tomato___healthy"]
    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction[0])
    predicted_class = classes[predicted_class_index]
    print(len(classes),prediction.shape)

    st.subheader(predicted_class)