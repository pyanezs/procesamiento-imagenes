# Libraries
import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
from pathlib import Path
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
    wd = os.path.join("outs", "uniforme")
    Path(wd).mkdir(parents=True, exist_ok=True)

    ########################################################################
    # Carga seccion de imagen
    img = load_section()
    print(img.shape)
    cv2.imshow("Imagen a utilizar", img)

    ########################################################################
    # Histograma Imagen
    counts, bins = np.histogram(img, bins=200, density=True)
    plt.figure()
    plt.title("Histograma Imagen")
    plt.hist(bins[:-1], bins, weights=counts)
    plt.savefig(os.path.join(wd, "img_hist.png"))
    plt.close('all')

    ########################################################################
    # Espectro Imagen
    img_fft = np.fft.fft2(img)
    spectrum = 0.1 * np.log(1 + np.abs(np.fft.fftshift(img_fft)))
    spectrum = cv2.normalize(spectrum, None, 0.0, 1.0, cv2.NORM_MINMAX)

    plt.figure()
    plt.title("Espectro imagen de entrada")
    plt.imshow(spectrum, cmap="gray")
    plt.savefig(os.path.join(wd, "img_spectrum.png"))
    plt.close('all')

    ########################################################################
    # Guarda seccion de imagen
    out_file = os.path.join(wd, "img.jpg")
    cv2.imwrite(out_file, img)

    ########################################################################
    # Agregar ruido
    img_norm = cv2.normalize(
        img.astype('float'),
        None,
        0.0,
        1.0,
        cv2.NORM_MINMAX)

    noisy_imgs = dict()

    for b in [40, 60, 150]:
        noise = np.random.uniform(low=10/255, high=b/255, size=img.shape)
        noisy = cv2.normalize((noise + img_norm), None, 0.0, 255.0, cv2.NORM_MINMAX)
        noisy = np.uint8(noisy)

        cv2.imshow(f"Imagen con ruido uniforme - b = {b}", noisy)
        out_file = os.path.join(wd, f"noisy_b{str(b).zfill(3)}.jpg")
        cv2.imwrite(out_file, noisy)
        noisy_imgs[b] = noisy

    ########################################################################
    # Filtrado
    # for i in range(5, 6):
    i = 5
    for var_n in [0.001, 0.003, 0.005, 0.01, 0.025, 0.03, 0.4]:
        def filtro_ruido_local(A):
            """Codigo entregado en clase"""
            var_N = var_n  # varianza estimada
            B = A.flatten()
            n = len(B)
            var_L = np.var(B)  # varianza en la mascara

            mu = np.mean(B)
            g = B[np.uint8(n/2)]
            f = g - (var_N/var_L)*(g-mu)
            return f

        for b, noisy in noisy_imgs.items():
            print(f"Filtrando imagen {b}")

            noisy_norm = cv2.normalize(noisy.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

            img = ndi.generic_filter(noisy_norm, filtro_ruido_local, [i, i])
            img = cv2.normalize(img, None, 0.0, 255.0, cv2.NORM_MINMAX)
            img = np.uint8(img)

            out_file = os.path.join(wd, f"filtered_{str(b).zfill(3)}_{str(var_n).zfill(5)}.jpg")
            cv2.imwrite(out_file, img)

    cv2.waitKey(0)


if __name__ == '__main__':
    exit(main(sys.argv))
