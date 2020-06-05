# Libraries
import os
import cv2
import sys

from pathlib import Path


def load_image():
    '''Carga imagen'''

    INPUT_FILE = os.path.join("inputs", "7062826349_4888c4f9d0_w.jpg")
    img = cv2.imread(INPUT_FILE, cv2.IMREAD_GRAYSCALE)
    return img


def load_section():
    '''Carga seccion de interes de la imagen y la normaliza'''
    img = load_image()

    x1 = 40
    x2 = min(img.shape[0], x1 + 256)

    y1 = 20
    y2 = min(img.shape[1], y1 + 256)

    return img[x1:x2, y1:y2]


def main(args):
    '''Main'''

    ########################################################################
    # Directorio de trabajo
    wd = os.path.join("outs", "gaussiano")
    Path(wd).mkdir(parents=True, exist_ok=True)

    ########################################################################
    # Carga seccion de imagen
    img = load_section()
    print(img.shape)
    cv2.imshow("Imagen a utilizar", img)

    ########################################################################
    # Guarda seccion de imagen
    out_file = os.path.join(wd, "img.jpg")
    cv2.imwrite(out_file, img)

    ########################################################################
    # Agregar ruido


    ########################################################################
    # Filtrar ruido





    cv2.waitKey(0)


if __name__ == '__main__':
    exit(main(sys.argv))
