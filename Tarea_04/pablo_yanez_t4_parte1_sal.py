# Libraries
import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage.util
from pathlib import Path
import scipy.ndimage as ndi


def load_image():
    '''Carga imagen'''

    INPUT_FILE = os.path.join("inputs", "dientes.jpg")
    img = cv2.imread(INPUT_FILE, cv2.IMREAD_GRAYSCALE)
    return img


def load_section():
    '''Carga seccion de interes de la imagen y la normaliza'''
    img = load_image()

    y1 = 20
    y2 = y1 + img.shape[0] - 1
    img = img[:-1, y1:y2]

    return img


def min_filter(roi):
    return np.min(roi.flatten())


def main(args):
    '''Main'''

    ########################################################################
    # Directorio de trabajo
    wd = os.path.join("outs", "sal")
    Path(wd).mkdir(parents=True, exist_ok=True)

    ########################################################################
    # Carga seccion de imagen
    img = load_section()
    cv2.imshow("Imagen a utilizar", img)

    ########################################################################
    # Guarda seccion de imagen
    out_file = os.path.join(wd, "img.jpg")
    cv2.imwrite(out_file, img)

    ########################################################################
    # Histograma Imagen
    counts, bins = np.histogram(img, bins=200)
    plt.figure()
    plt.title("Histograma Imagen")
    plt.hist(bins[:-1], bins, weights=counts)
    plt.savefig(os.path.join(wd, "img_hist.png"))
    plt.close('all')

    ########################################################################
    # Agregar ruido
    noisy = skimage.util.random_noise(
        img,
        mode='s&p',
        seed=0,
        amount=0.08,
        salt_vs_pepper=1)

    noisy = cv2.normalize(noisy.astype('float'), None, 0.0, 255, cv2.NORM_MINMAX)
    noisy = np.uint8(noisy)

    cv2.imshow("Imagen con ruido sal", noisy)
    out_file = os.path.join(wd, "noisy.jpg")
    cv2.imwrite(out_file, noisy)

    ########################################################################
    # # Sal y pimienta para informe
    # noisy = skimage.util.random_noise(
    #     img,
    #     mode='s&p',
    #     seed=0,
    #     amount=0.08,
    #     salt_vs_pepper=0.5)
    # noisy = np.uint8(noisy * 255)

    # out_file = os.path.join(wd, "noisy_sp.jpg")
    # cv2.imwrite(out_file, noisy)

    ########################################################################
    # Histograma Imagen con Rudio
    counts, bins = np.histogram(noisy, bins=200)
    plt.figure()
    plt.title("Histograma Imagen con Ruido")
    plt.hist(bins[:-1], bins, weights=counts)
    plt.savefig(os.path.join(wd, "noisy_hist.png"))
    plt.close('all')

    ########################################################################
    # Filtrar ruido
    masks = [(i, j) for i in range(1, 4) for j in range(1, 4)]

    for i, j in masks:
        filtered = ndi.generic_filter(noisy, min_filter, [i, j])
        out_file = os.path.join(wd, f"filtered_{i}x{j}.jpg")
        cv2.imwrite(out_file, filtered)
        cv2.imshow(f"Imagen Filtrada - min() - {i}x{j}", filtered)

    cv2.waitKey(0)


if __name__ == '__main__':
    exit(main(sys.argv))
