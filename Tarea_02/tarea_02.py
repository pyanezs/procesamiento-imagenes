# Libraries
import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# GLOBAL VARIABLES
OUTPUT_DIR = "outs"
INPUT_FILE = "fallas.tif"


# Custom Functions
def put_text(text, img, text_color=(0, 0, 0)):
    # https://stackoverflow.com/questions/16615662/how-to-write-text-on-a-image-in-windows-using-python-opencv2
    new_image = img.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
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
    img = cv2.imread(INPUT_FILE, cv2.IMREAD_GRAYSCALE)
    return img


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
    labels = ["Blue", "Green", "Red"]

    for text, ch in zip(labels, median):
        cv2.imshow(f"Ch: {text} Median: {size}", ch)

    # Save image for document
    display = [put_text(text, ch) for text, ch in zip(labels, median)]
    display = np.hstack(display)
    save_img(f"median_{size}.jpg", display)


def main(args):
    '''Main'''

    # Create ouput dir
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # if len(args) != 2:
    #     print("Wrong usage. Call using one of the following options:")
    #     for i in range(1, 7):
    #         print(f" * P{i}")
    #     print(" * ALL")
    #     exit(1)

    ##########################################################################
    # P1
    img_fallas = load_image()
    cv2.imshow("Imagen entrada", img_fallas)
    save_img('input_image.jpg', img_fallas)



    cv2.waitKey(0)


if __name__ == '__main__':
    exit(main(sys.argv))

