# Libraries
import os
import cv2
import gdown
import numpy as np

from matplotlib import pyplot as plt
from pathlib import Path

# GLOBAL VARIABLES
OUTPUT_DIR = "outs"
INPUT_FILE = "portrait.jpg"
OUPUT_PREFIX = "portrait"

# Custom Functions
def put_text(text, img, text_color=(0,0,0)):
    # https://stackoverflow.com/questions/16615662/how-to-write-text-on-a-image-in-windows-using-python-opencv2
    new_image = img.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,30)
    fontScale = 1
    lineType = 2

    cv2.putText(
        new_image,
        text,
        bottomLeftCornerOfText,
        font,
        fontScale,
        text_color,
        lineType)

    return new_image


def save_img(filename, img):
    out_file = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(out_file, img)


def download_image():
    # Image was downloaded from the following url.
    # https://www.pexels.com/photo/women-s-white-and-black-button-up-collared-shirt-774909/
    # For simplicity file was uploaded to Google Drive so is easier to reproduce
    # the results from the code
    url = "https://drive.google.com/uc?id=1kEp1hqSjFi_YDlFflEXPuU_G0cWANdXk"
    gdown.download(url, INPUT_FILE, quiet=True)


def load_and_show():
    portrait = cv2.imread(INPUT_FILE)
    cv2.imshow("Original Portrait", portrait)
    return portrait


def display_channels(img):
    channels = cv2.split(img)

    # Put text to identify each image
    text = ["Blue", "Green", "Red"]
    channels_text = [put_text(text,ch) for text,ch in zip(text, channels)]

    # Combine un single array
    channels = np.hstack(channels)
    channels_text = np.hstack(channels_text)

    # Display image
    cv2.imshow('Image Channels', channels_text)

    # Save image to file
    save_img(f"{OUPUT_PREFIX}_bgr.jpg", channels)
    save_img(f"{OUPUT_PREFIX}_bgr_text.jpg", channels_text)


def channels_histogram(img):

    plt.style.use('fast')
    plt.title('Histograma por canal')

    # Show graph
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()

    # Save graph
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])

    output_file = os.path.join(OUTPUT_DIR, f"{OUPUT_PREFIX}_hist")
    plt.savefig(output_file)


def equalize_channels(img):

    channels = cv2.split(img)

    channels_eq = list()
    for ch in channels:
        channels_eq.append(cv2.equalizeHist(ch))

    # Put text to identify each image
    text = ["Blue", "Green", "Red"]
    channels_eq_text = [put_text(text,ch) for text,ch in zip(text, channels_eq)]

    channels_eq_text = np.hstack(channels_eq_text)

    # Display and save image
    cv2.imshow('Portrait Channels Equalized', channels_eq_text)
    save_img(f"{OUPUT_PREFIX}_bgr_eq_text.jpg", channels_eq_text)



# channels_eq = np.hstack([portrait_blue_eq, portrait_green_eq, portrait_red_eq])
# cv2.imshow('Portrait Channels Equalized', channels_eq)
# cv2.imwrite(f"{out_prefix}_bgr_eq.jpg", channels_eq)

# #%% Compare
# channels_compare = np.vstack([channels, channels_eq])

# cv2.imshow('Portrait Channels: Before and After', channels_compare)

def main():
    # Create ouput dir
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    download_image()

    # Q1
    portrait = load_and_show()

    # Q2
    display_channels(portrait)

    # Q3
    channels_histogram(portrait)

    equalize_channels(portrait)

    cv2.waitKey(0)



if __name__ == '__main__':
    main()