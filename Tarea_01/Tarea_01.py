# Libraries
import os
import cv2
import sys
import numpy as np

from pathlib import Path

# GLOBAL VARIABLES
OUTPUT_DIR = "outs"
INPUT_FILE = "portrait.jpg"


# Custom Functions
def put_text(text, img, text_color=(0, 0, 0)):
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


def load_image():
    portrait = cv2.imread(INPUT_FILE)
    return portrait


def gamma_correction(img, factor):
    """Simple function that applies gamma function to a image"""
    # Scale image
    img = img / 255.0

    # Applies pow function to each pixel
    img = cv2.pow(img, factor)

    # De-scale and return and numpy.uint8
    return np.uint8(img * 255)


def apply_median(img, size):

    channels = cv2.split(img)
    median = [cv2.medianBlur(ch, size) for ch in channels]
    text = ["Blue", "Green", "Red"]

    for text,ch in zip(text, median):
        cv2.imshow(f"Ch: {text} Median: {size}", ch)

    # Save image for document
    display = [put_text(text,ch) for text,ch in zip(text, median)]
    display = np.hstack(display)
    save_img(f"median_{size}.jpg", display)


def q1():
    img = load_image()
    cv2.imshow(f"Retrato", img)


def q2():
    img = load_image()

    channels = cv2.split(img)
    text = ["Blue", "Green", "Red"]

    # Display one window per channel
    for text, ch in zip(text, channels):
        cv2.imshow(f'Canal {text}', ch)

    # Export single image for text
    # Put text to identify each image
    channels = [put_text(text,ch) for text,ch in zip(text, channels)]
    channels = np.hstack(channels)
    save_img(f"bgr.jpg", channels)


def q3():
    img = load_image()

    channels = cv2.split(img)
    text = ["Blue", "Green", "Red"]

    channels_eq = list()
    for ch in channels:
        channels_eq.append(cv2.equalizeHist(ch))

    # Display one window per channel
    for text, ch in zip(text, channels):
        cv2.imshow(f'Canal {text} Ecualizado', ch)

    # Export single image for text document
    # Put text to identify each image
    channels_eq = [put_text(text,ch) for text,ch in zip(text, channels_eq)]
    channels_eq = np.hstack(channels_eq)
    save_img(f"bgr_eq.jpg", channels_eq)


def q4():
    img = load_image()

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

    # After manually inspecting the previous result the following gammas were
    # selected
    gamma_per_channel = [0.8, 0.9, 1.2]

    # Split image per channels
    channels = cv2.split(img)

    img_gamma = [
        gamma_correction(ch, factor)
        for ch, factor in zip(channels, gamma_per_channel)]

    text = ["Blue", "Green", "Red"]

    for index in range(0, 3):
        cv2.imshow(f'Canal {text[index]} | Factor Gamma: {gamma_per_channel[index]}', img_gamma[index])

    # Export single image for text document

    text = [
        f"B|g={gamma_per_channel[0]}",
        f"G|g={gamma_per_channel[1]}",
        f"R|g={gamma_per_channel[2]}"]

    # Add text
    img_gamma_text = [put_text(text,ch) for text,ch in zip(text, img_gamma)]
    display = np.hstack(img_gamma_text)

    save_img(f"gamma.jpg", display)


def q5():
    img = load_image()
    portrait_media_3 = apply_median(img, 3)
    portrait_media_5 = apply_median(img, 5)


def q6():
    img = load_image()
    channels = cv2.split(img)

    # Equalize channels
    channels_eq = list()
    for ch in channels:
        channels_eq.append(cv2.equalizeHist(ch))

    # Apply gamma correctionn
    gamma_per_channel = [0.8, 0.9, 1.2]
    channels_gamma = [
        gamma_correction(ch, factor)
        for ch, factor in zip(channels, gamma_per_channel)]

    # Median filter
    channels_median_3 = [cv2.medianBlur(ch, 3) for ch in channels]
    channels_median_5 = [cv2.medianBlur(ch, 5) for ch in channels]

    # Merge channles of each process
    img1 = cv2.merge(channels_eq)
    img2 = cv2.merge(channels_gamma)
    img3 = cv2.merge(channels_median_3)
    img4 = cv2.merge(channels_median_5)

    img_12 = cv2.addWeighted(img1, 0.25, img2, 0.25, 0)
    img_34 = cv2.addWeighted(img3, 0.25 ,img4, 0.25, 0)

    output_image = cv2.add(img_12, img_34)
    cv2.imshow(f"Retrato - Combinacion de 4 Imagenes", output_image)


def main(args):
    # Create ouput dir
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    if len(args) != 2:
        print("Wrong usage. Call using one of the following options:")
        for i in range(1, 7):
            print(f" * P{i}")
        print(f" * ALL")
        exit(1)

    args[1] = args[1].upper()
    # Main program
    if args[1] == "P1":
        q1()
    elif args[1] == "P2":
        q2()
    elif args[1] == "P3":
        q3()
    elif args[1] == "P4":
        q4()
    elif args[1] == "P5":
        q5()
    elif args[1] == "P6":
        q6()
    elif args[1] == "ALL":
        q1()
        q2()
        q3()
        q4()
        q5()
        q6()
    else:
        print("Wrong usage. Call using one of the following options:")
        for i in range(1, 7):
            print(f" * P{i}")

    cv2.waitKey(0)


if __name__ == '__main__':
    exit(main(sys.argv))
