# Libraries
import os
import cv2
import sys
import numpy as np
import numpy.matlib

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

    out_file = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(out_file, img)


def load_image():
    '''Carga imagen'''
    img = cv2.imread(INPUT_FILE, cv2.IMREAD_GRAYSCALE)
    return img


def load_section():
    '''Carga seccion de interes de la imagen y la normaliza'''
    img = load_image()
    h_offset = 220
    return img[0:512, h_offset:(h_offset + 512)]


def get_image_spectrum(img):
    '''Returns normalized spectrum'''

    fft_img = np.fft.fft2(img)
    fshift = np.fft.fftshift(fft_img)
    spectrum = np.log(np.abs(fshift))

    spectrum = cv2.normalize(
        spectrum,
        None,
        0.0,
        1.0,
        cv2.NORM_MINMAX)

    return spectrum


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
    save_img("puente.jpg", img)

    # Espectro imagen
    spectrum = get_image_spectrum(img)
    cv2.imshow('Image spectrum', spectrum)

    # Guarda archivo con espectro
    spectrum = cv2.normalize(
        spectrum.astype('float'),
        None,
        0,
        256,
        cv2.NORM_MINMAX)
    spectrum = np.uint8(spectrum)
    save_img("puente_fft.jpg", spectrum)


def gen_noise(freq, dim):
    '''Genera ruido de la frecuencia y tamaño especficado'''
    values = np.linspace(0, 1, num=dim)
    noise = 0.2 * np.sin(2 * np.pi * freq * values)
    return np.matlib.repmat(noise, dim, 1)


def pregunta_2():
    '''P2: Agrega ruido a las imagenes'''
    freqs = [10, 50, 80]

    for freq in freqs:
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
        save_img(f"puente_f{freq}.jpg", img)

        # Espectro imagen
        spectrum = get_image_spectrum(img)
        cv2.imshow('Image spectrum', spectrum)

        # Guarda archivo con espectro
        spectrum = cv2.normalize(
            spectrum.astype('float'),
            None,
            0,
            256,
            cv2.NORM_MINMAX)
        spectrum = np.uint8(spectrum)
        save_img(f"puente_f{freq}_fft.jpg", spectrum)


def main(args):
    '''Main'''
    # Directorio de salida
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    pregunta_1()

    pregunta_2()


    #############################
    # freqs = [10, 50, 80]

    # for freq in freqs:
    #     # Carga y normaliza imagen
    #     img = load_section()
    #     img = cv2.normalize(
    #         img.astype('float'),
    #         None,
    #         0.0,
    #         1.0,
    #         cv2.NORM_MINMAX)

    #     # Genera y aplica ruido
    #     noise = gen_noise(freq=freq, dim=512)
    #     img = np.add(img, noise)

    #     imgs.append(img)
    #     cv2.imshow(f"Imagen con ruido | f: {freq}[Hz]", img)

    #     # Denormaliza y guarda a archivo
    #     img = cv2.normalize(img.astype('float'), None, 0, 256, cv2.NORM_MINMAX)
    #     img = np.uint8(img)

    #     # Filtrado imagen

    #     fa = 10  # Frecuencia de corte
    #     fc = 28  # Frecuencia de corte
    #     fb = 30  # Frecuencia de corte
    #     N = 3  # orden del filtro

    #     x = np.linspace(-2*fc, 2*fc, 256)
    #     y = np.linspace(-2*fc, 2*fc, 256)
    #     X, Y = np.meshgrid(x, y)
    #     H1 = 1 / (1+(np.power(np.sqrt(X**2+Y**2)/fa, 2*N)))
    #     H2 = 1 - 1 / (1+(np.power(np.sqrt(X**2+Y**2)/fb, 2*N)))
    #     Z = H1 * H2




    #############################


    cv2.waitKey()

    # Display image

    # if len(args) != 2:
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
