# Libraries
import os
import cv2
import sys
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools

from pathlib import Path

# GLOBAL VARIABLES
OUTPUT_DIR = "outs_2"
INPUT_FILE = "cameraman.png"

# Custom Functions
def load_image():
    '''Carga imagen'''
    img = cv2.imread(INPUT_FILE, cv2.COLOR_BGR2GRAY)
    return img


def show_save_spectrum(spectrum, title, out_file, save_only=False, cmap="gray"):
    """Muestra y guarda archivo con espectro usando matplotlib"""
    fig = plt.figure()
    plt.title(title)
    plt.imshow(spectrum, cmap=cmap)
    plt.savefig(out_file)
    plt.close('all')

    if save_only:
        return

    fig = plt.figure()
    plt.title(title)
    plt.imshow(spectrum, cmap=cmap)
    plt.show()


def butterworh_lp(X, Y, fc, n, X0=0, Y0=0):
    '''Filtro Butterworh Pasabajos '''

    return (1 / (1 + (np.power(np.sqrt((X - X0) ** 2 + (Y - Y0) ** 2) / fc, 2 * n))))


def get_modified_image():
    '''Añade el ruido indicado en el enunciado'''
    img = load_image()

    m = img.shape[0]
    delta = 15
    V = np.fix(np.linspace(delta, m - delta, delta)).astype('uint8')
    img[V, :] = img[V, :] + 50
    img[:, V] = img[:, V] + 50

    return img


def pregunta_1():
    '''P1: Anade ruido indicado'''
    wd = os.path.join(OUTPUT_DIR, "p1")
    Path(wd).mkdir(parents=True, exist_ok=True)

    # Seccion a utilizar
    img = load_image()
    cv2.imshow("Cameraman", img)

    # Imagen modificada
    img = get_modified_image()
    cv2.imshow("Cameraman - Modificado", img)
    out_file = os.path.join(wd, "cameraman_mod.jpg")
    cv2.imwrite(out_file, img)

    # Espectro imagen
    fft_img = np.fft.fft2(img)
    fshift = np.fft.fftshift(fft_img)
    spectrum = 0.1 * np.log(1000 + np.abs(fshift))
    spectrum = cv2.normalize(spectrum, None, 0.0, 1.0, cv2.NORM_MINMAX)

    show_save_spectrum(
        spectrum,
        f'Espectro Imagen Modificada',
        os.path.join(wd, "cameraman_mod_fft.png"))

    # Filtado - Pasabajos Fc = 5 Hz


def pregunta_2():

    wd = os.path.join(OUTPUT_DIR, "p2")
    Path(wd).mkdir(parents=True, exist_ok=True)

    # Carga imagen
    img = get_modified_image()

    # FFT imagen
    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)

    # Spectrum
    spectrum = 0.1 * np.log(100 + np.abs(img_fft))
    spectrum = cv2.normalize(spectrum, None, 0.0, 1.0, cv2.NORM_MINMAX)

    cv2.imshow(f"Spectrum", spectrum)

    # Valores usados para inspeccion
    # thresholds = [(i, j)
    #     for i in range(50, 90, 5)
    #     for j in range(i + 25, 90, 5)]
    thresholds = [(50, 75)]

    for lower_thresh, upper_thresh in thresholds:
        lower_thresh = lower_thresh / 100
        upper_thresh = upper_thresh / 100

        # Threshold Valores con alta energia
        rt, spectrum_bina_1 = cv2.threshold(spectrum, lower_thresh, 1.0, cv2.THRESH_BINARY)
        rt, spectrum_bina_2 = cv2.threshold(spectrum, upper_thresh, 1.0, cv2.THRESH_BINARY)

        title = f"Spectrum - Binary Thesh = {lower_thresh}"
        out_file = os.path.join(wd, f"spect_bina_{lower_thresh:.2f}.png")
        show_save_spectrum(spectrum_bina_1, title, out_file,save_only=True, cmap="gray")

        title = f"Spectrum - Binary Thesh = {upper_thresh}"
        out_file = os.path.join(wd, f"spect_bina_{upper_thresh:.2f}.png")
        show_save_spectrum(spectrum_bina_2, title, out_file,save_only=True, cmap="gray")

        # Tiene valor 1 en todos los puntos en el espectro que hay que atenuar
        mask = 1 - (spectrum_bina_1 - spectrum_bina_2)
        cv2.imshow(f"Mask", mask)

        title = f"Mask Thresh LT: {lower_thresh} UT: {upper_thresh:.2f}"
        out_file = os.path.join(wd, f"{lower_thresh:.2f}_{upper_thresh:.2f}_mask.png")

        show_save_spectrum(mask, title, out_file, save_only=True, cmap="gray")

        for factor in range(0,1):
            factor = factor / 10

            my_filter = mask
            my_filter[my_filter < 1] = factor

            # Aplicacion filtro
            new_img_fft = my_filter * img_fft

            # Guarda espectro
            spectrum = 0.1 * np.log(100 + np.abs(new_img_fft))
            spectrum = cv2.normalize(spectrum, None, 0.0, 1.0, cv2.NORM_MINMAX)
            spectrum = np.uint8(spectrum * 255)

            show_save_spectrum(
                spectrum,
                f'Espectro Imagen Filtrada',
                os.path.join(wd, f"{lower_thresh: .2f}_{upper_thresh: .2f}_cm_{factor: .2f}_spectrum.png"),
                save_only=False)

            # FFT Inversa
            new_img = np.fft.ifft2(np.fft.fftshift(new_img_fft))
            new_img = cv2.normalize(
                abs(new_img),
                None,
                0.0,
                1.0,
                cv2.NORM_MINMAX)

            new_img = np.uint8(new_img * 255)

            out_file = os.path.join(
                wd, f"{lower_thresh:.2f}_{upper_thresh:.2f}_cm_{factor:.2f}.png")

            cv2.imwrite(out_file, new_img)

            cv2.imshow("Imagen Filtrada", new_img)


def main(args):
    '''Main'''

    pregunta_1()

    pregunta_2()

    cv2.waitKey(0)


if __name__ == '__main__':
    exit(main(sys.argv))
