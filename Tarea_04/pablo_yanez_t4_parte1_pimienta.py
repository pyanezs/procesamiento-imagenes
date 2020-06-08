# Libraries
import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage.util
from pathlib import Path

import mediana_adaptivo as ma

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
    wd = os.path.join("outs", "pimienta")
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
    plt.savefig(os.path.join(wd, f"img_hist.png"))
    plt.close('all')

    ########################################################################
    # Espectro Imagen
    img_fft = np.fft.fft2(img)
    spectrum = 0.1 * np.log(1 + np.abs(np.fft.fftshift(img_fft)))
    spectrum = cv2.normalize(spectrum, None, 0.0, 1.0, cv2.NORM_MINMAX)

    fig = plt.figure()
    plt.title("Espectro imagen de entrada")
    plt.imshow(spectrum, cmap="gray")
    plt.savefig(os.path.join(wd, f"img_spectrum.png"))
    plt.close('all')

    ########################################################################
    # Agregar ruido
    noisy = skimage.util.random_noise(img, mode='pepper', seed=1, amount=0.20)
    noisy = np.uint8(noisy * 255)

    cv2.imshow("Imagen con ruido pimienta", noisy)
    out_file = os.path.join(wd, "noisy.jpg")
    cv2.imwrite(out_file, noisy)

    ########################################################################
    # Histograma Imagen con Rudio
    counts, bins = np.histogram(noisy, bins=200)
    plt.figure()
    plt.title("Histograma Imagen con Ruido")
    plt.hist(bins[:-1], bins, weights=counts)
    plt.savefig(os.path.join(wd, f"noisy_hist.png"))
    plt.close('all')

    ########################################################################
    # Espectro Imagen con Ruido
    noisy_fft = np.fft.fft2(noisy)
    spectrum = 0.1 * np.log(1 + np.abs(np.fft.fftshift(noisy_fft)))
    spectrum = cv2.normalize(spectrum, None, 0.0, 1.0, cv2.NORM_MINMAX)

    fig = plt.figure()
    plt.title("Espectro imagen con Ruido")
    plt.imshow(spectrum, cmap="gray")
    plt.savefig(os.path.join(wd, f"noisy_spectrum.png"))
    plt.close('all')

    ########################################################################
    # Filtrar ruido

    img_filter = ma.filtro_mediana_adaptiva(img)
    cv2.imshow("Imagen filtrada", noisy)

    cv2.waitKey(0)


if __name__ == '__main__':
    exit(main(sys.argv))
