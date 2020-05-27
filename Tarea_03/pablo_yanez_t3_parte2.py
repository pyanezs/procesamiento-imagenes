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
    spectrum = np.log(np.abs(fshift))
    spectrum = cv2.normalize(spectrum, None, 0.0, 1.0, cv2.NORM_MINMAX)

    show_save_spectrum(
        spectrum,
        f'Espectro Imagen Modificada',
        os.path.join(wd, "cameraman_mod_fft.png"),
        cmap=cm.Spectral)

    # Filtado - Pasabajos Fc = 5 Hz


def pregunta_2_sol1():

    wd = os.path.join(OUTPUT_DIR, "p2")
    Path(wd).mkdir(parents=True, exist_ok=True)

    # Carga imagen
    img = get_modified_image()

    # FFT imagen
    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)

    # Mesh para los filtros
    dim = img.shape[0]
    X, Y = np.meshgrid(
        np.linspace(-dim/2, dim/2, dim),
        np.linspace(-dim/2, dim/2, dim))

    for fc in range(1, 20, 1):
        low_pass = butterworh_lp(X, Y, fc, 3)

        # Aplicacion filtro
        new_img_fft = low_pass * img_fft

        # FFT Inversa
        new_img = np.fft.ifft2(np.fft.fftshift(new_img_fft))
        new_img = cv2.normalize(
            abs(new_img),
            None,
            0.0,
            1.0,
            cv2.NORM_MINMAX)

        new_img = np.uint8(new_img * 255)

        cv2.imshow(f"Cameraman - Filtrado fc: {fc}", new_img)

        out_file = os.path.join(
            wd, f"fc_{str(fc).zfill(2)}.png")
        cv2.imwrite(out_file, new_img)


def pregunta_2_sol2():

    wd = os.path.join(OUTPUT_DIR, "p2a")
    Path(wd).mkdir(parents=True, exist_ok=True)

    # Carga imagen
    img = get_modified_image()

    # FFT imagen
    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)

    # Spectrum
    spectrum = np.log(np.abs(img_fft))
    spectrum = cv2.normalize(spectrum, None, 0.0, 1.0, cv2.NORM_MINMAX)

    # Threshold Valores con alta energia
    rt, spectrum_bina_1 = cv2.threshold(spectrum, 0.6, 1.0, cv2.THRESH_BINARY)
    rt, spectrum_bina_2 = cv2.threshold(spectrum, 0.8, 1.0, cv2.THRESH_BINARY)

    cv2.imshow(f"Binary Spectrum 1", spectrum_bina_1)
    cv2.imshow(f"Binary Spectrum 2", spectrum_bina_2)

    # Tiene valor 1 en todos los puntos en el espectro que hay que atenuar
    mask = 1 - (spectrum_bina_1 - spectrum_bina_2)
    cv2.imshow(f"Mask", mask)

    for factor in range(0,11):
        factor = factor / 10
        my_filter = mask
        my_filter[my_filter < 1] = factor

        # Aplicacion filtro
        new_img_fft = my_filter * img_fft

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
            wd, f"cameraman_factor_{str(factor).zfill(3)}_0.png")

        cv2.imwrite(out_file, new_img)

        new_img = cv2.medianBlur(new_img, 3)
        out_file = os.path.join(
            wd, f"cameraman_factor_{str(factor).zfill(3)}_1.png")
        cv2.imwrite(out_file, new_img)




def main(args):
    '''Main'''

    options = ["ALL", "P1", "P2-A", "P2-B"]

    if len(args) != 2 or args[1].upper() not in options:
        print("Wrong usage. Call using one of the following options:")
        for option in options:
            print(f" * {option}")
        exit(1)

    # Convertir entrada a mayusculas
    args[1] = args[1].upper()

    # Main program
    if args[1] in ["P1", "ALL"]:
        pregunta_1()

    if args[1] in ["P2-A", "ALL"]:
        pregunta_2_sol1()

    if args[1] in ["P2-B", "ALL"]:
        pregunta_2_sol2()


    cv2.waitKey(0)


if __name__ == '__main__':
    exit(main(sys.argv))
