# Libraries
import os
import cv2
import sys
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib import cm

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


def save_img(filename, img):
    '''Guarda imagen a archivo en carpeta de salida'''

    out_file = f"{OUTPUT_DIR}/{filename}"
    cv2.imwrite(out_file, img)


def load_image():
    '''Carga imagen'''
    img = cv2.imread(INPUT_FILE, cv2.COLOR_BGR2GRAY)
    return img


def load_section():
    '''Carga seccion de interes de la imagen y la normaliza'''
    img = load_image()
    h_offset = 220
    return img[0:512, h_offset:(h_offset + 512)]


def get_image_fft_and_spectrum(img):
    '''Returns normalized spectrum of image'''

    fft_img = np.fft.fft2(img)
    fshift = np.fft.fftshift(fft_img)
    spectrum = np.log(np.abs(fshift))

    spectrum = cv2.normalize(
        spectrum,
        None,
        0.0,
        1.0,
        cv2.NORM_MINMAX)

    return fshift, spectrum


def image_0_to_256(img):
    spectrum = cv2.normalize(
        img.astype('float'),
        None,
        0,
        256,
        cv2.NORM_MINMAX)
    return np.uint8(spectrum)


def pregunta_1():
    '''P1: Despliega imagen seleccionada'''
    # Imagen original
    img = load_image()
    cv2.imshow("Imagen", img)
    print(f"Dimensiones imagen {img.shape}")

    # Seccion a utilizar
    img = load_section()
    cv2.imshow("Area Seleccionada", img)
    print(f"Dimensiones seccion a utilizar {img.shape}")
    save_img("p1/puente.jpg", img)

    # Espectro imagen
    _, spectrum = get_image_fft_and_spectrum(img)
    spectrum = np.uint8(spectrum * 255)

    cv2.imshow('Image spectrum', spectrum)

    # Guarda archivo con espectro
    save_img("p1/puente_fft.jpg", spectrum)


def gen_noise(freq, dim):
    '''Genera ruido de la frecuencia y tamaño especficado'''
    values = np.linspace(0, 1, num=dim)
    noise = 0.1 * np.sin(2 * np.pi * freq * values)
    return np.matlib.repmat(noise, dim, 1)


def pregunta_2():
    '''P2: Agrega ruido a las imagenes'''
    freqs = [10, 50, 80]

    for freq in freqs:

        # Carpeta de resultados
        Path(f"{OUTPUT_DIR}/p2/noise_{freq}").mkdir(parents=True, exist_ok=True)

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
        save_img(f"/p2/noise_{freq}/puente.jpg", img)

        # Espectro imagen
        fft, spectrum = get_image_fft_and_spectrum(img)
        spectrum = np.uint8(spectrum * 255)
        cv2.imshow('Image spectrum', spectrum)

        # Guarda archivo con espectro
        save_img(f"/p2/noise_{freq}/puente_fft.jpg", spectrum)


def butterworh_lp(X, Y, fc, n):
    '''Filtro Butterworh'''
    return (1 / (1 + (np.power(np.sqrt(X ** 2 + Y ** 2) / fc, 2 * n))))


def pregunta_3():
    freqs = [10, 50, 80]

    filter_order = 3
    for freq in freqs:

        Path(f"{OUTPUT_DIR}/p3/noise_{freq}").mkdir(parents=True, exist_ok=True)

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

        # Dimensio imagen: Se conoce que es cuadrada
        dim = img.shape[0]

        # Mesh para los filtros
        X, Y = np.meshgrid(
            np.linspace(-dim/2, dim/2, dim),
            np.linspace(-dim/2, dim/2, dim))

        # Espectro imagen
        img_fft, _ = get_image_fft_and_spectrum(img)

        # Se prueba filtros de distintos ordenes
        for filter_order in range(1, 25, 2):

            # Primer Filtro: Pasabajos invertido -> High Pass
            high_pass = 1 - butterworh_lp(X, Y, freq - 1, filter_order)

            # Segundo Filtro: Low-Pass
            low_pass = butterworh_lp(X, Y, freq + 1, filter_order)

            # Mezclar ambos filtros -> Band-stop
            band_stop = 1 - high_pass * low_pass
            band_stop = band_stop - np.min(band_stop)
            band_stop = band_stop / np.max(band_stop)

            img_filtered_ftt = band_stop * img_fft
            spectrum = 0.1*np.log(1 + np.abs(img_filtered_ftt))
            spectrum = np.uint8(spectrum * 255)

            save_img(
                f"p3/noise_{freq}/puente_fft_N{str(filter_order).zfill(2)}.jpg",
                spectrum)

            img_filtered = np.fft.ifft2(np.fft.fftshift(img_filtered_ftt))
            img_filtered = cv2.normalize(
                abs(img_filtered),
                None,
                0.0,
                1.0,
                cv2.NORM_MINMAX)

            cv2.imshow(
                f"Imagen Filtrada | f: {freq}[Hz] | N: {filter_order}",
                img_filtered)

            img_filtered = np.uint8(img_filtered * 255)
            save_img(
                f"p3/noise_{freq}/puente_N{str(filter_order).zfill(2)}.jpg",
                img_filtered)


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
    # Directorio de salida
    Path(f"{OUTPUT_DIR}/p1").mkdir(parents=True, exist_ok=True)
    Path(f"{OUTPUT_DIR}/p2").mkdir(parents=True, exist_ok=True)
    Path(f"{OUTPUT_DIR}/p3").mkdir(parents=True, exist_ok=True)
    Path(f"{OUTPUT_DIR}/p4").mkdir(parents=True, exist_ok=True)

    # pregunta_1()
    # pregunta_2()
    # pregunta_3()

    #############################
    # Pregunta 4
    img, img_orig = get_modified_image()
    cv2.imshow('Imagen con ruido segun instrucciones', img)
    save_img("p4/img_noise.jpg", img)

    # Espectro y fft de la imagen
    img_fft, spectrum = get_image_fft_and_spectrum(img)
    spectrum = np.uint8(spectrum * 255)

    cv2.imshow('Espectro imagen con ruido', spectrum)
    save_img("p4/img_noise_fft.jpg", spectrum)

    # Diffence of spectrums
    img_fft_orig, spectrum_orig = get_image_fft_and_spectrum(img_orig)

    diff = cv2.normalize(
        abs(- img_fft_orig + img_fft),
        None,
        0.0,
        1.0,
        cv2.NORM_MINMAX)

    print(diff)
    cv2.imshow('Diffs ffts', diff)

    print(diff)
    # cv2.imshow('Diff spectrums', diff)







    # Display image

    # if len    (args) != 2:
    #     print("Wrong usage. Call using one of the following options:")
    #     for i in range(1, 7):
    #         print(f" * P{i}")
    #     print(" * ALL")
    #     exit(1)

    # args[1] = args[1].upper()
    # # Main program
    # if args[1] == "P1":
    #     pregunta_1()
    # elif args[1] == "P2":
    #     pregunta_2()
    # elif args[1] == "P3":
    #     pregunta_3()
    # elif args[1] == "P4":
    #     pregunta_4()
    # elif args[1] == "P5":
    #     pregunta_5()
    # elif args[1] == "P6":
    #     pregunta_6()
    # elif args[1] == "ALL":
    #     pregunta_1()
    #     pregunta_2()
    #     pregunta_3()
    #     pregunta_4()
    #     pregunta_5()
    #     pregunta_6()
    # else:
    #     print("Wrong usage. Call using one of the following options:")
    #     for i in range(1, 7):
    #         print(f" * P{i}")

    cv2.waitKey(0)


if __name__ == '__main__':
    exit(main(sys.argv))
