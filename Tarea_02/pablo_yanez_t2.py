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
SELECTED_MEDIAN_MASK = 77
SELECTED_THRESHOLD = 10


# Custom Functions
def put_text(text, img, text_color=(0, 0, 0)):
    # https://stackoverflow.com/questions/16615662/how-to-write-text-on-a-image-in-windows-using-python-opencv2
    new_image = img.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 320)
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


def pregunta_1():
    img_fallas = load_image()
    cv2.imshow("Imagen entrada", img_fallas)
    save_img('input_image.jpg', img_fallas)


def pregunta_2():
    img_fallas = load_image()

    # Hist
    counts, bins = np.histogram(img_fallas, bins=250)
    plt.figure()
    plt.hist(bins[:-1], bins, weights=counts)
    plt.savefig(os.path.join(OUTPUT_DIR, f"hist_fallas.png"))

    # Equalize
    img_fallas_eq = cv2.equalizeHist(img_fallas)
    # Hist
    counts, bins = np.histogram(img_fallas_eq, bins=250)
    plt.figure()
    plt.hist(bins[:-1], bins, weights=counts)
    plt.savefig(os.path.join(OUTPUT_DIR, f"hist_fallas_eq.png"))

    cv2.imshow("P2: Imagen Equalizada", img_fallas_eq)
    save_img('input_image_eq.jpg', img_fallas_eq)


def pregunta_3():

    img_fallas = load_image()
    r_offset = 80
    img_fallas = img_fallas[:, (635-r_offset):(968-r_offset)]

    for mask in range(3, 80, 2):
        img_median = cv2.medianBlur(img_fallas, mask)
        img_median = put_text(
            f"{mask}x{mask}", img_median, text_color=(255, 0, 0))
        save_img(f'median_filter_{mask}.jpg', img_median)

    img_fallas = load_image()
    img_median = cv2.medianBlur(img_fallas, SELECTED_MEDIAN_MASK)
    cv2.imshow(
        f"P3: Filtro Mediana {SELECTED_MEDIAN_MASK}x{SELECTED_MEDIAN_MASK}",
        img_median)
    save_img(f'median_filter_{SELECTED_MEDIAN_MASK}_completo.jpg', img_median)



def pregunta_4():

    img_fallas = load_image()
    img_fallas_eq = cv2.equalizeHist(img_fallas)
    img_median = cv2.medianBlur(img_fallas, SELECTED_MEDIAN_MASK)

    img = cv2.subtract(img_median, img_fallas_eq)

    # Apply pseudocolor
    # https://www.learnopencv.com/applycolormap-for-pseudocoloring-in-opencv-c-python/
    img_pseudocolor = cv2.applyColorMap(img, cv2.COLORMAP_JET)

    # Show images
    cv2.imshow("Original", img_fallas)
    cv2.imshow("P4", img)
    cv2.imshow("P4 - Pseudocolor", img_pseudocolor)

    # Save file
    save_img(f'img_p4.jpg', img)
    save_img(f'img_p4_color.jpg', img_pseudocolor)


def pregunta_5():
    img_fallas = load_image()
    img_fallas_eq = cv2.equalizeHist(img_fallas)
    img_median = cv2.medianBlur(img_fallas, SELECTED_MEDIAN_MASK)

    img = cv2.subtract(img_median, img_fallas_eq)

    r_offset = 80
    img = img[:, (635-r_offset):(968-r_offset)]

    for value in range(1, 55, 1):
        rt, bina = cv2.threshold(img, value, 255, cv2.THRESH_BINARY)

        bina = put_text(f"Threshold: {value}", bina, text_color=(255, 255, 255))

        save_img(f'img_p5_threshold_{value}.jpg', bina)

    img = cv2.subtract(img_median, img_fallas_eq)
    rt, bina = cv2.threshold(img, SELECTED_THRESHOLD, 255, cv2.THRESH_BINARY)
    cv2.imshow(f"P5: Threshold {SELECTED_THRESHOLD}", bina)


def pregunta_6():
    img_fallas = load_image()
    img_fallas_eq = cv2.equalizeHist(img_fallas)
    img_median = cv2.medianBlur(img_fallas, SELECTED_MEDIAN_MASK)

    img_p4 = cv2.subtract(img_median, img_fallas_eq)

    rt, bina = cv2.threshold(img_p4, SELECTED_THRESHOLD, 255, cv2.THRESH_BINARY)
    bina = np.uint8(bina / 255.0)

    img_p6 = cv2.multiply(bina, img_fallas_eq)

    cv2.imshow(f"P6: Imagen final", img_p6)
    save_img(f'img_final.jpg', img_p6)



def main(args):
    '''Main'''

    # Create ouput dir
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    if len(args) != 2:
        print("Wrong usage. Call using one of the following options:")
        for i in range(1, 7):
            print(f" * P{i}")
        print(" * ALL")
        exit(1)

    args[1] = args[1].upper()
    # Main program
    if args[1] == "P1":
        pregunta_1()
    elif args[1] == "P2":
        pregunta_2()
    elif args[1] == "P3":
        pregunta_3()
    elif args[1] == "P4":
        pregunta_4()
    elif args[1] == "P5":
        pregunta_5()
    elif args[1] == "P6":
        pregunta_6()
    elif args[1] == "ALL":
        pregunta_1()
        pregunta_2()
        pregunta_3()
        pregunta_4()
        pregunta_5()
        pregunta_6()
    else:
        print("Wrong usage. Call using one of the following options:")
        for i in range(1, 7):
            print(f" * P{i}")


    cv2.waitKey(0)


if __name__ == '__main__':
    exit(main(sys.argv))

