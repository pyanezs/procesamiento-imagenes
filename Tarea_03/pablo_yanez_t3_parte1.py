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
OUTPUT_DIR = "outs"
INPUT_FILE = "puente.jpg"

# Custom Functions
def put_text(text, img, text_color=(0, 0, 0)):
    '''Inserta texto en una imagen'''
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


def load_image():
    '''Carga imagen'''
    img = cv2.imread(INPUT_FILE, cv2.COLOR_BGR2GRAY)
    return img


def load_section():
    '''Carga seccion de interes de la imagen y la normaliza'''
    img = load_image()
    h_offset = 220
    return img[0:512, h_offset:(h_offset + 512)]


def show_save_spectrum(spectrum, title, out_file, save_only=False, cmap = "gray"):

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


def pregunta_1():
    '''P1: Despliega imagen seleccionada'''
    wd = os.path.join(OUTPUT_DIR, "p1")

    # Imagen original
    img = load_image()
    cv2.imshow("Imagen", img)
    print(f"Dimensiones imagen {img.shape}")

    # Seccion a utilizar
    img = load_section()
    cv2.imshow("Area Seleccionada", img)
    print(f"Dimensiones seccion a utilizar {img.shape}")
    out_file = os.path.join(wd, "puente_roi.jpg")
    cv2.imwrite(out_file, img)

    # Espectro imagen
    fft_img = np.fft.fft2(img)
    fshift = np.fft.fftshift(fft_img)
    spectrum = np.log(np.abs(fshift))
    spectrum = cv2.normalize(spectrum, None, 0.0, 1.0, cv2.NORM_MINMAX)
    spectrum = np.uint8(spectrum * 255)

    show_save_spectrum(
        spectrum,
        f'Espectro Imagen',
        os.path.join(wd, "puente_fft.png"),
        cmap=cm.Spectral)


def gen_noise(freq, dim):
    '''Genera ruido de la frecuencia y tamaño especficado'''
    values = np.linspace(0, 1, num=dim)
    noise = 0.1 * np.sin(2 * np.pi * freq * values)
    return np.matlib.repmat(noise, dim, 1)


def pregunta_2():
    '''P2: Agrega ruido a las imagenes'''
    for freq in [10, 50, 80]:
        wd = os.path.join(OUTPUT_DIR, "p2", str(freq))
        # Carpeta de resultados
        Path(wd).mkdir(parents=True, exist_ok=True)

        # Carga y normaliza imagen
        img = load_section()
        img = cv2.normalize(
            img.astype('float'),
            None,
            0.0,
            1.0,
            cv2.NORM_MINMAX)

        # Genera y aplica ruido
        noise = gen_noise(freq=freq, dim=512)
        img = np.add(img, noise)
        cv2.imshow(f"Imagen con ruido | f: {freq}[Hz]", img)

        # Denormaliza y guarda a archivo
        img = cv2.normalize(img.astype('float'), None, 0, 256, cv2.NORM_MINMAX)
        img = np.uint8(img)
        out_file = os.path.join(wd, "puente.jpg")
        cv2.imwrite(out_file, img)

        # Espectro imagen
        fft_img = np.fft.fft2(img)
        fshift = np.fft.fftshift(fft_img)
        spectrum = np.log(np.abs(fshift))

        show_save_spectrum(
            spectrum,
            f'Ruido {freq} Hz',
            os.path.join(wd, "puente_fft.png"),
            cmap = cm.Spectral)


def butterworh_lp(X, Y, fc, n, X0=0, Y0=0):
    '''Filtro Butterworh'''
    return (1 / (1 + (np.power(np.sqrt((X - X0) ** 2 + (Y - Y0) ** 2) / fc, 2 * n))))


def band_stop_filter(X, Y, fc1, fc2, n, X0 = 0, Y0 = 0):

    # Primer Filtro: Pasabajos invertido -> High Pass
    high_pass = 1 - butterworh_lp(X, Y, fc1, n, X0, Y0)

    # Segundo Filtro: Low-Pass
    low_pass = butterworh_lp(X, Y, fc2, n, X0, Y0)

    # Mezclar ambos filtros -> Band-stop
    band_stop = 1 - high_pass * low_pass
    band_stop = band_stop - np.min(band_stop)
    band_stop = band_stop / np.max(band_stop)

    return band_stop


def pregunta_3():

    freq = [10, 50, 80]
    filter_order = [x for x in range(1, 25, 2)]

    data = [freq, filter_order]
    data = list(itertools.product(*data))

    for freq, filter_order in data:
        wd = os.path.join(OUTPUT_DIR, "p3", str(freq))
        Path(wd).mkdir(parents=True, exist_ok=True)

        # Carga y normaliza imagen
        img = load_section()
        img = cv2.normalize(img.astype('float'), None,
                            0.0, 1.0, cv2.NORM_MINMAX)

        # Genera y aplica ruido
        noise = gen_noise(freq=freq, dim=512)
        img = np.add(img, noise)

        cv2.imshow(f"Imagen con ruido | f: {freq}[Hz]", img)

        # Dimension imagen: Se conoce que es cuadrada
        dim = img.shape[0]

        # Mesh para los filtros
        X, Y = np.meshgrid(
            np.linspace(-dim/2, dim/2, dim),
            np.linspace(-dim/2, dim/2, dim))

        # FFT imagen
        img_fft = np.fft.fft2(img)
        img_fft = np.fft.fftshift(img_fft)

        band_stop = band_stop_filter(X, Y, freq-1, freq+1, filter_order)

        # Aplicacion filtro
        new_img_fft = band_stop * img_fft

        new_img = np.fft.ifft2(np.fft.fftshift(new_img_fft))
        new_img = cv2.normalize(
            abs(new_img),
            None,
            0.0,
            1.0,
            cv2.NORM_MINMAX)

        new_img = np.uint8(new_img * 255)

        out_file = os.path.join(
            wd, f"puente_N{str(filter_order).zfill(2)}.png")
        cv2.imwrite(out_file, new_img)


def get_modified_image():
    '''Añade el ruido indicado en el enunciado'''

    img = load_section()
    img_orig = img.copy()

    m = img.shape[0]
    delta = 15
    V = np.fix(np.linspace(delta, m - delta, delta)).astype('uint8')
    img[V, :] = img[V, :] + 50
    img[:, V] = img[:, V] + 50

    return img, img_orig


def main(args):
    '''Main'''

    if len(args) != 2:
        print("Wrong usage. Call using one of the following options:")
        for i in range(1, 4):
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
    elif args[1] == "ALL":
        pregunta_1()
        pregunta_2()
        pregunta_3()
    else:
        print("Wrong usage. Call using one of the following options:")
        for i in range(1, 4):
            print(f" * P{i}")

    cv2.waitKey(0)


if __name__ == '__main__':
    exit(main(sys.argv))
