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


def filtro_mediana_adaptiva(roi, s_max=16, ws=4):
    '''Codigo de material de clase'''
    m, n = roi.shape
    J = np.zeros([m, n])

    count = 0
    for i in range(0, m - s_max):
        for j in range(0, n - s_max):
            count += 1
            sw = True
            Zxy = roi[i, j]
            wadapt = ws
            while sw:
                B = roi[i:i+wadapt, j:j+wadapt]
                pixel, newwin = bloque(B, Zxy, wadapt, s_max)
                if (pixel != -1):
                    J[i, j] = pixel
                    sw = False
                else:
                    wadapt = newwin

    return (np.uint8(J[0:(m-s_max), 0:(n-s_max)]))


def bloque(A, Zxy, wadapt, Smax):

    newwin = wadapt

    Zmin = np.min(A.flatten())
    Zmax = np.max(A.flatten())
    Zmed = np.median(A.flatten())
    px = -1

    A1 = Zmed - Zmin
    A2 = Zmed - Zmax

    if (newwin <= Smax):
        #% Nivel A
        if (A1 > 0) & (A2 < 0):
            #%nivel B
            B1 = float(Zxy)-float(Zmin)
            B2 = float(Zxy)-float(Zmax)

            if (B1 > 0) & (B2 < 0):
                px = Zxy
            else:
                px = Zmed
        else:
            # Incrementamos el tamaño de
            # la ventana
            newwin = newwin+1
    else:
        px = Zxy

    return(px, newwin)



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

        # noisy = np.uint8(cv2.multiply(noise, img_norm) * 255)
        cv2.imshow(f"Imagen con ruido uniforme - b = {b}", noisy)
        out_file = os.path.join(wd, f"noisy_b{str(b).zfill(3)}.jpg")
        cv2.imwrite(out_file, noisy)
        noisy_imgs[b] = noisy

    ########################################################################
    # Filtrado

    for b, noisy in noisy_imgs.items():
        print(f"Filtrando imagen {b}")
        img = filtro_mediana_adaptiva(noisy)
        out_file = os.path.join(wd, f"filtered_{str(b).zfill(3)}.jpg")
        cv2.imwrite(out_file, img)

    cv2.waitKey(0)


if __name__ == '__main__':
    exit(main(sys.argv))
