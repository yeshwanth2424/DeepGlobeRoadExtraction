import streamlit as st
import numpy as np
import cv2 as cv
from PIL import Image
from Custom_UNet_model import build_unet


def extract_road_path(model, image):

    # predict the pixel probability
    predicted_proba = model.predict(image[np.newaxis, :, :, :])
    predict_mask = np.argmax(predicted_proba, axis=-1)

    # Replace 1 with 255 and 0 remains as it is because [0:background, 255:road]
    predict_img = np.where(predict_mask == 1, 255, 0)

    # split the channels
    red, green, blue = cv.split(image)

    # detecting road path with red colour so red is highlighted and other channels are neutralized
    red = red + predict_img[0]
    red = np.where((red > 255), 255, red)

    green = green + predict_img[0]
    green = np.where((green > 255), 0, green)

    blue = blue + predict_img[0]
    blue = np.where((blue > 255), 0, blue)

    # concat the channels back after red colour is highlighted
    final_prediction = np.dstack((red, green, blue))

    return final_prediction


def main():

    # giving a title
    st.title('DEEP GLOBE ROAD EXTRACTION')

    # read the image using file_uploader component
    image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])

    # create two columns to display satellite image and prediction image
    original,predict = st.columns(2)

    # prediction
    if image_file is not None:
        open_img = Image.open(image_file)

        # display original image
        with original:
            st.header("Satellite Image")
            st.image(open_img,width=330)

        # preprocess and resize image
        img = np.asarray(open_img)
        size = 256
        img = cv.resize(img,(size,size))

        # create model and load the pretrained weights into it
        model = build_unet((256, 256, 3))
        model.load_weights('weights-24-0.765.h5')

        # Extract Road path
        if st.button("Extract Road path"):
            predicted_image = extract_road_path(model,img)
            with predict:
                st.header("Road Path Prediction")
                st.image(predicted_image,width=330)


if __name__ == '__main__':
    main()