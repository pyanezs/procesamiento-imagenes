# Libraries
import os
import cv2
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import skimage.util
import scipy.ndimage as ndi
from math import pi


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


def main(args):
    '''Main'''

    ########################################################################
    # Directorio de trabajo
    wd = os.path.join("outs", "gaussiano")
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
    noisy_imgs = dict()
    for i in [5, 10, 50]:
        sigma = i/100
        noisy = skimage.util.random_noise(
            img,
            mode="gaussian",
            var=sigma ** 2,
            seed=0)
        noisy = np.uint8(noisy * 255)

        cv2.imshow("Imagen con ruido gaussiano", noisy)
        out_file = os.path.join(wd, f"noisy_{str(i).zfill(3)}.jpg")
        cv2.imwrite(out_file, noisy)

        noisy_imgs[i] = noisy
    ########################################################################
    # Histograma Imagen con Rudio
    for i, noisy in noisy_imgs.items():
        noisy = noisy_imgs[i]

        counts, bins = np.histogram(noisy, bins=200)
        plt.figure()
        plt.title("Histograma Imagen con Ruido")
        plt.hist(bins[:-1], bins, weights=counts)
        plt.savefig(os.path.join(wd, f"noisy_hist_{str(i).zfill(3)}.png"))
        plt.close('all')

    ########################################################################
    # Filtrar ruido
    for i in range(3, 8):
        print(f"Tamaño ventana: {i}")

        # Filtro de codigos de clase
        def filtro_gaussiano(roi):
            roi = np.reshape(roi, (i, i))
            sigma = 2.2
            t = i
            ventana = np.linspace(-t/2, t/2, t)
            u, v = np.meshgrid(ventana, ventana)
            G = (1/(sigma**2*2*pi))*np.exp(-(u**2+v**2)/(2*sigma**2))
            N = G/np.sum(G.flatten())  # normalizamos
            T = N * roi
            return np.sum(T)

        for index, noisy in noisy_imgs.items():
            print("Filtrando imagen")
            filtered = ndi.generic_filter(noisy, filtro_gaussiano, [i, i])
            out_file = os.path.join(wd, f"filtered_{index}_{i}x{i}.jpg")
            cv2.imwrite(out_file, filtered)
            # cv2.imshow(f"Imagen Filtrada - media - {i}x{i}", filtered)

    cv2.waitKey(0)


if __name__ == '__main__':
    exit(main(sys.argv))
