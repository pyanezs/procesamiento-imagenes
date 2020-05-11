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
    channels = [put_text(text,ch) for text,ch in zip(text, channels)]

    # Combine un single array
    channels = np.hstack(channels)

    # Display image
    cv2.imshow('Image Channels', channels)

    # Save image to file
    save_img(f"bgr.jpg", channels)


def channels_histogram(img, file):

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

    output_file = os.path.join(OUTPUT_DIR, f"{file}_hist")
    plt.savefig(output_file)


def equalize_channels(img):

    channels = cv2.split(img)

    channels_eq = list()
    for ch in channels:
        channels_eq.append(cv2.equalizeHist(ch))

    img_eq = cv2.merge(channels_eq)

    # Put text to identify each image
    text = ["Blue", "Green", "Red"]
    channels_eq = [put_text(text,ch) for text,ch in zip(text, channels_eq)]

    channels_eq = np.hstack(channels_eq)

    # Display and save image
    cv2.imshow('Portrait Channels Equalized', channels_eq)
    save_img(f"bgr_eq.jpg", channels_eq)

    return img_eq


def gamma_correction(img, factor):
    """Simple function that applies gamma function to a image"""
    # Scale image
    img = img / 255.0

    # Applies pow function to each pixel
    img = cv2.pow(img, factor)

    # De-scale and return and numpy.uint8
    return np.uint8(img * 255)


def apply_multiple_gammas(img):

    # Split image per channels
    channels = cv2.split(img)

    # List of gammas to try
    gammas = [x/10 for x in range(5,20)]

    gamma_per_channel = {"B": [], "G": [], "R": []}

    for factor in gammas:
        # Apply gamma correction to each channel
        img_gamma = [gamma_correction(ch, factor) for ch in channels]
        img_gamma = [put_text(f"g={factor}", ch) for ch in img_gamma]

        gamma_per_channel["B"].append(img_gamma[0])
        gamma_per_channel["G"].append(img_gamma[1])
        gamma_per_channel["R"].append(img_gamma[2])

    for channel, gamma in gamma_per_channel.items():

        line_1 = np.hstack(gamma[0:5])
        line_2 = np.hstack(gamma[5:10])
        line_3 = np.hstack(gamma[10:15])

        display = np.vstack([line_1, line_2, line_3])

        save_img(f"gamma_{channel}.jpg", display)


def apply_gamma_per_channel(img, gamma_per_channel):
    # Split image per channels
    channels = cv2.split(img)

    img_gamma = [
        gamma_correction(ch, factor)
        for ch, factor in zip(channels, gamma_per_channel)]

    text = [
        f"B|g={gamma_per_channel[0]}",
        f"G|g={gamma_per_channel[1]}",
        f"R|g={gamma_per_channel[2]}"]

    # Add text
    img_gamma_text = [put_text(text,ch) for text,ch in zip(text, img_gamma)]
    display = np.hstack(img_gamma_text)
    cv2.imshow(f"Gamma Correction", display)
    save_img(f"gamma.jpg", display)

    return cv2.merge(img_gamma)


def main():
    # Create ouput dir
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    ####################################################################
    # Q1: Download image and load using opencv
    download_image()
    portrait = load_and_show()

    ####################################################################
    # Q2: Display image channels
    display_channels(portrait)

    ####################################################################
    # Q3: Equalize histograms
    channels_histogram(portrait, "orig")
    portrait_eq = equalize_channels(portrait)
    channels_histogram(portrait_eq, "eq")

    ####################################################################
    # Q4: Apply gamma

    # Apply a gamma function with a range of values.
    # Results are written per channen on file
    apply_multiple_gammas(portrait)

    # After seeing the results of the previous step
    # we choose the following factors for each of the channels
    b_factor = 0.8
    g_factor = 0.9
    r_factor = 1.2

    portrait_gamma = apply_gamma_per_channel(portrait, [b_factor, g_factor, r_factor])

    cv2.imshow(f"Portrait - Post Gamma Correction", portrait_gamma)
    save_img(f"portrait_gamma.jpg", portrait_gamma)

    ####################################################################
    # Q5:

    ####################################################################
    # Q6:



    cv2.waitKey(0)



if __name__ == '__main__':
    main()