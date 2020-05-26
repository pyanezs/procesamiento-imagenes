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


def show_save_spectrum(spectrum, title, out_file, save_only=False, cmap="gray"):

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


def get_modified_image():
    '''Añade el ruido indicado en el enunciado'''
    img = load_section()

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
    img = load_section()
    cv2.imshow("Area Seleccionada", img)

    # Espectro imagen
    fft_img = np.fft.fft2(img)
    fshift = np.fft.fftshift(fft_img)
    spectrum = np.log(np.abs(fshift))
    spectrum = cv2.normalize(spectrum, None, 0.0, 1.0, cv2.NORM_MINMAX)

    show_save_spectrum(
        spectrum,
        f'Espectro Imagen',
        os.path.join(wd, "puente_fft.png"),
        cmap=cm.Spectral)


    # Imagen modificada
    img = get_modified_image()
    cv2.imshow("Area Seleccionada - Modifiada", img)
    out_file = os.path.join(wd, "puente_roi_mod.jpg")
    cv2.imwrite(out_file, img)

    # Espectro imagen
    fft_img = np.fft.fft2(img)
    fshift = np.fft.fftshift(fft_img)
    spectrum = np.log(np.abs(fshift))
    spectrum = cv2.normalize(spectrum, None, 0.0, 1.0, cv2.NORM_MINMAX)

    show_save_spectrum(
        spectrum,
        f'Espectro Imagen Modificada',
        os.path.join(wd, "puente_mod_fft.png"),
        cmap=cm.Spectral)


def gen_noise(freq, dim):
    '''Genera ruido de la frecuencia y tamaño especficado'''
    values = np.linspace(0, 1, num=dim)
    noise = 0.1 * np.sin(2 * np.pi * freq * values)
    return np.matlib.repmat(noise, dim, 1)


def pregunta_2():

    wd = os.path.join(OUTPUT_DIR, "p2")
    Path(wd).mkdir(parents=True, exist_ok=True)

    # Carga imagen
    img = get_modified_image()

    imgs = [
        img[0:256, 0:256],
        img[0:256, 256:],
        img[256:, 0:256],
        img[256:, 256:],
        img
    ]

    # Histograma espectro
    fft_img = np.fft.fft2(imgs[2])
    fft_shift = np.fft.fftshift(fft_img)
    spectrum = np.log(np.abs(fft_shift))
    spectrum = cv2.normalize(spectrum, None, 0.0, 1.0, cv2.NORM_MINMAX)

    # fig = plt.figure()
    # plt.title("Spectrum")
    # plt.imshow(spectrum, cmap=cm.Spectral)
    # plt.show()

    # # Comentado para evitar salida en pantalla
    # counts, bins = np.histogram(spectrum, bins=200)
    # plt.figure()
    # plt.hist(bins[:-1], bins, weights=counts)
    # plt.show()

    # plt.savefig(os.path.join(OUTPUT_DIR, f"hist_fallas.png")

    # Binarizacion espectro

    # for i in range(50,72,1):
    rt, bina1 = cv2.threshold(spectrum, 0.6, 1.0, cv2.THRESH_BINARY)
    rt, bina2 = cv2.threshold(spectrum, 0.725, 1.0, cv2.THRESH_BINARY)

    mask = 1 - (bina1 - bina2)
    cv2.imshow("Bina1", bina1)
    cv2.imshow("Bina2", bina2)
    cv2.imshow("Mask", mask)

    # Se usa mascara para atenuar frecuencias de interes
    # factors = [x/10 for x in range(0,10)]

    mask = 1 - (bina1 - bina2)
    mask[mask < 1.0] = 0

    fft_new_img = mask * fft_shift
    new_img = np.fft.ifft2(np.fft.fftshift(fft_new_img))
    new_img = cv2.normalize(
        abs(new_img),
        None,
        0.0,
        1.0,
        cv2.NORM_MINMAX)

    # new_img = np.uint8(new_img * 255)
    cv2.imshow(
        f"Aplicacion filtro |", new_img)

pass


def main(args):
    '''Main'''

    # pregunta_1()
    pregunta_2()
    cv2.waitKey(0)
    exit()


    if len(args) != 2:
        print("Wrong usage. Call using one of the following options:")
        for i in range(1, 3):
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



# # Libraries
# import os
# import cv2
# import sys
# import numpy as np
# import numpy.matlib
# import matplotlib.pyplot as plt
# from matplotlib import cm
# import itertools

# from pathlib import Path

# # GLOBAL VARIABLES
# OUTPUT_DIR = "outs"
# INPUT_FILE = "puente.jpg"

# # Custom Functions
# def put_text(text, img, text_color=(0, 0, 0)):
#     '''Inserta texto en una imagen'''
#     # https://stackoverflow.com/questions/16615662/how-to-write-text-on-a-image-in-windows-using-python-opencv2
#     new_image = img.copy()

#     font = cv2.FONT_HERSHEY_SIMPLEX
#     bottomLeftCornerOfText = (10, 320)
#     fontScale = 1
#     lineType = 2

#     cv2.putText(
#         new_image,
#         text,
#         bottomLeftCornerOfText,
#         font,
#         fontScale,
#         text_color,
#         lineType)

#     return new_image


# def save_img(filename, img):
#     '''Guarda imagen a archivo en carpeta de salida'''

#     out_file = f"{OUTPUT_DIR}/{filename}"
#     cv2.imwrite(out_file, img)


# def load_image():
#     '''Carga imagen'''
#     img = cv2.imread(INPUT_FILE, cv2.COLOR_BGR2GRAY)
#     return img


# def load_section():
#     '''Carga seccion de interes de la imagen y la normaliza'''
#     img = load_image()
#     h_offset = 220
#     return img[0:512, h_offset:(h_offset + 512)]


# def get_image_fft_and_spectrum(img):
#     '''Returns normalized spectrum of image'''

#     fft_img = np.fft.fft2(img)
#     fshift = np.fft.fftshift(fft_img)
#     spectrum = np.log(0.1 * np.abs(100 + 0.1 * fshift))

#     spectrum = cv2.normalize(
#         spectrum,
#         None,
#         0.0,
#         1.0,
#         cv2.NORM_MINMAX)

#     return fshift, spectrum


# def image_0_to_256(img):
#     spectrum = cv2.normalize(
#         img.astype('float'),
#         None,
#         0,
#         256,
#         cv2.NORM_MINMAX)
#     return np.uint8(spectrum)






# def pregunta_2():
#     '''P2: Agrega ruido a las imagenes'''
#     freqs = [10, 50, 80]

#     for freq in freqs:

#         # Carpeta de resultados
#         Path(f"{OUTPUT_DIR}/p2/noise_{freq}").mkdir(parents=True, exist_ok=True)

#         # Carga y normaliza imagen
#         img = load_section()
#         img = cv2.normalize(
#             img.astype('float'),
#             None,
#             0.0,
#             1.0,
#             cv2.NORM_MINMAX)

#         # Genera y aplica ruido
#         noise = gen_noise(freq=freq, dim=512)
#         img = np.add(img, noise)
#         cv2.imshow(f"Imagen con ruido | f: {freq}[Hz]", img)

#         # Denormaliza y guarda a archivo
#         img = cv2.normalize(img.astype('float'), None, 0, 256, cv2.NORM_MINMAX)
#         img = np.uint8(img)
#         save_img(f"/p2/noise_{freq}/puente.jpg", img)

#         # Espectro imagen
#         fft, spectrum = get_image_fft_and_spectrum(img)
#         spectrum = np.uint8(spectrum * 255)
#         cv2.imshow('Image spectrum', spectrum)

#         # Guarda archivo con espectro
#         save_img(f"/p2/noise_{freq}/puente_fft.jpg", spectrum)

#         fig = plt.figure()
#         plt.imshow(spectrum, cmap="gray")
#         plt.show()


# def butterworh_lp(X, Y, fc, n, X0=0, Y0=0):
#     '''Filtro Butterworh'''
#     return (1 / (1 + (np.power(np.sqrt((X - X0) ** 2 + (Y - Y0) ** 2) / fc, 2 * n))))


# def band_stop_filter(X, Y, fc1, fc2, n, X0 = 0, Y0 = 0):

#     # Primer Filtro: Pasabajos invertido -> High Pass
#     high_pass = 1 - butterworh_lp(X, Y, fc1, n, X0, Y0)

#     # Segundo Filtro: Low-Pass
#     low_pass = butterworh_lp(X, Y, fc2, n, X0, Y0)

#     # Mezclar ambos filtros -> Band-stop
#     band_stop = 1 - high_pass * low_pass
#     band_stop = band_stop - np.min(band_stop)
#     band_stop = band_stop / np.max(band_stop)

#     return band_stop


# def pregunta_3():

#     freq = [10, 50, 80]
#     filter_order = [x for x in range(1, 25, 2)]

#     data = [freq, filter_order]
#     data = list(itertools.product(*data))

#     for freq, filter_order in data:

#         Path(f"{OUTPUT_DIR}/p3/noise_{freq}").mkdir(parents=True, exist_ok=True)

#         # Carga y normaliza imagen
#         img = load_section()
#         img = cv2.normalize(
#             img.astype('float'),
#             None,
#             0.0,
#             1.0,
#             cv2.NORM_MINMAX)

#         # Genera y aplica ruido
#         noise = gen_noise(freq=freq, dim=512)
#         img = np.add(img, noise)

#         cv2.imshow(f"Imagen con ruido | f: {freq}[Hz]", img)

#         # Dimensio imagen: Se conoce que es cuadrada
#         dim = img.shape[0]

#         # Mesh para los filtros
#         X, Y = np.meshgrid(
#             np.linspace(-dim/2, dim/2, dim),
#             np.linspace(-dim/2, dim/2, dim))

#         # Espectro imagen
#         img_fft, _ = get_image_fft_and_spectrum(img)

#         band_stop = band_stop_filter(X, Y, freq-1, freq+1, filter_order)

#         img_filtered_ftt = band_stop * img_fft
#         spectrum = 0.1*np.log(1 + np.abs(img_filtered_ftt))
#         spectrum = np.uint8(spectrum * 255)

#         save_img(
#             f"p3/noise_{freq}/puente_fft_N{str(filter_order).zfill(2)}.jpg",
#             spectrum)

#         img_filtered = np.fft.ifft2(np.fft.fftshift(img_filtered_ftt))
#         img_filtered = cv2.normalize(
#             abs(img_filtered),
#             None,
#             0.0,
#             1.0,
#             cv2.NORM_MINMAX)

#         cv2.imshow(
#             f"Imagen Filtrada | f: {freq}[Hz] | N: {filter_order}",
#             img_filtered)

#         img_filtered = np.uint8(img_filtered * 255)
#         save_img(
#             f"p3/noise_{freq}/puente_N{str(filter_order).zfill(2)}.jpg",
#             img_filtered)


# def get_modified_image():
#     '''Añade el ruido indicado en el enunciado'''

#     img = load_section()
#     img_orig = img.copy()

#     m = img.shape[0]
#     delta = 15
#     V = np.fix(np.linspace(delta, m - delta, delta)).astype('uint8')
#     img[V, :] = img[V, :] + 50
#     img[:, V] = img[:, V] + 50

#     return img, img_orig


# def main(args):
#     '''Main'''
#     # Directorio de salida
#     Path(f"{OUTPUT_DIR}/p1").mkdir(parents=True, exist_ok=True)
#     Path(f"{OUTPUT_DIR}/p2").mkdir(parents=True, exist_ok=True)
#     Path(f"{OUTPUT_DIR}/p3").mkdir(parents=True, exist_ok=True)
#     Path(f"{OUTPUT_DIR}/p4").mkdir(parents=True, exist_ok=True)

#     pregunta_1()
#     # pregunta_2()
#     # pregunta_3()

#     #############################
#     # Pregunta 4

#     # fw = 2
#     # img_noise, img_orig = get_modified_image()

#     # fft_noise, spectrum_noise = get_image_fft_and_spectrum(img_noise)
#     # fft_orig, spectrum_orig = get_image_fft_and_spectrum(img_orig)

#     # cv2.imshow(
#     #     f"spectrum_noise",
#     #     spectrum_noise)

#     # diff = cv2.normalize(
#     #     np.log(np.abs(fft_noise - fft_orig)),
#     #     None,
#     #     0.0,
#     #     1.0,
#     #     cv2.NORM_MINMAX)

#     # fig = plt.figure()
#     # plt.imshow(diff, cmap="gray")
#     # plt.title(f'Diff Full')
#     # plt.show()

#     # save_img("p4/puente.jpg", img_noise)

#     # ################################################
#     # # Dividir en dos con un filtro ideal
#     # dim = img_noise.shape[0]
#     # X, Y = np.meshgrid(
#     #     np.linspace(-dim/2, dim/2, dim),
#     #     np.linspace(-dim/2, dim/2, dim))

#     # LOW_PASS = np.sqrt(X**2+Y**2) < fw
#     # HIGH_PASS = np.sqrt(X**2+Y**2) >= fw

#     # ################################################
#     # img_low = fft_noise * LOW_PASS

#     # img = np.fft.ifft2(np.fft.fftshift(img_low))
#     # img = cv2.normalize(
#     #     abs(img),
#     #     None,
#     #     0.0,
#     #     1.0,
#     #     cv2.NORM_MINMAX)

#     # cv2.imshow(
#     #     f"Imagen Baja Frecuencia",
#     #     img)

#     # diff = cv2.normalize(
#     #     np.abs(img_low - fft_orig),
#     #     None,
#     #     0.0,
#     #     1.0,
#     #     cv2.NORM_MINMAX)

#     # ################################################
#     # img_high = fft_noise * HIGH_PASS

#     # img = np.fft.ifft2(np.fft.fftshift(img_high))
#     # img = cv2.normalize(
#     #     abs(img),
#     #     None,
#     #     0.0,
#     #     1.0,
#     #     cv2.NORM_MINMAX)

#     # cv2.imshow(
#     #     f"Imagen Alta Frecuencia",
#     #     img)

#     # # # Componentes de alta frecuencia
#     # h_pass = 1 - butterworh_lp(X, Y, 10, 3)
#     # img_high = fft_noise * h_pass

#     # spectrum = np.log(np.abs(img_high))
#     # spectrum = cv2.normalize(
#     #     spectrum,
#     #     None,
#     #     0.0,
#     #     1.0,
#     #     cv2.NORM_MINMAX)

#     # fig = plt.figure()
#     # plt.imshow(spectrum, cmap="gray")
#     # plt.title(f'High-Freq Img')
#     # plt.show()

#     # img_filtered = np.fft.ifft2(np.fft.fftshift(img_high))
#     # img_filtered = cv2.normalize(
#     #     abs(img_filtered),
#     #     None,
#     #     0.0,
#     #     1.0,
#     #     cv2.NORM_MINMAX)

#     # cv2.imshow(
#     #     f"Alta Frecuencia",
#     #     img_filtered)

#     # # Componentes de baja fecuencia
#     # l_pass = butterworh_lp(X, Y, 11, 3)
#     # img_low = fft_noise * l_pass

#     # spectrum = np.log(np.abs(img_low))
#     # spectrum = cv2.normalize(
#     #     spectrum,
#     #     None,
#     #     0.0,
#     #     1.0,
#     #     cv2.NORM_MINMAX)

#     # fig = plt.figure()
#     # plt.imshow(spectrum, cmap="gray")
#     # plt.title(f'Low-Freq Img')
#     # plt.show()

#     # img_filtered = np.fft.ifft2(np.fft.fftshift(img_low))
#     # img_filtered = cv2.normalize(
#     #     abs(img_filtered),
#     #     None,
#     #     0.0,
#     #     1.0,
#     #     cv2.NORM_MINMAX)

#     # cv2.imshow(
#     #     f"Low Frecuencia",
#     #     img_filtered)

#     # Filtro que elimina frecuencias en el eje X
#     # x = np.linspace(-dim/2, dim/2, dim)

#     # values = list()
#     # for value in x:
#     #     if value < -fw:
#     #         values.append(1)
#     #     elif value > fw:
#     #         values.append(1)
#     #     elif value < 0:
#     #         values.append(-value/fw)
#     #     else:
#     #         values.append(value/fw)

#     # values = np.array(values)
#     # M = np.matlib.repmat(values, dim, 1)

#     # fig = plt.figure()
#     # plt.imshow(M, cmap="gray")
#     # plt.title(f'Custom filter - Y')
#     # plt.show()

#     # img_high_filtered = img_high * M

#     # img_high_filtered = img_high_filtered * np.transpose(M)

#     # img_fft = img_high_filtered + img_low

#     # img_filtered = np.fft.ifft2(np.fft.fftshift(img_fft))
#     # img_filtered = cv2.normalize(
#     #     abs(img_filtered),
#     #     None,
#     #     0.0,
#     #     1.0,
#     #     cv2.NORM_MINMAX)

#     # cv2.imshow(
#     #     f"Imagen Filtrada",
#     #     img_filtered)



#     # Filtar

#     # cv2.imshow(f"LP", l_pass)
#     # cv2.imshow(f"HP", h_pass)

#     # # Idea pasar seccion a

#     # img = img_noise

#     # l_pass = butterworh_lp(X, Y, fc1, 3)
#     # h_pass =



#     # fig = plt.figure()
#     # ax = fig.add_subplot(111, projection='3d')
#     # ax.plot_surface(X, Y, Z, cmap=cm.Spectral)
#     # plt.show()




#     # cv2.imshow('Imagen con ruido segun instrucciones', img)
#     # save_img("p4/img_noise.jpg", img)

#     # # Espectro y fft de la imagen
#     # img_fft, spectrum = get_image_fft_and_spectrum(img)
#     # spectrum = np.uint8(spectrum * 255)

#     # cv2.imshow('Espectro imagen con ruido', spectrum)
#     # save_img("p4/img_noise_fft.jpg", spectrum)

#     # # Diffence of spectrums
#     # img_fft_orig, spectrum_orig = get_image_fft_and_spectrum(img_orig)

#     # diff = cv2.normalize(
#     #     abs(- img_fft_orig + img_fft),
#     #     None,
#     #     0.0,
#     #     1.0,
#     #     cv2.NORM_MINMAX)

#     # print(diff)
#     # cv2.imshow('Diffs ffts', diff)

#     # print(diff)
#     # cv2.imshow('Diff spectrums', diff)







#     # Display image

#     # if len    (args) != 2:
#     #     print("Wrong usage. Call using one of the following options:")
#     #     for i in range(1, 7):
#     #         print(f" * P{i}")
#     #     print(" * ALL")
#     #     exit(1)

#     # args[1] = args[1].upper()
#     # # Main program
#     # if args[1] == "P1":
#     #     pregunta_1()
#     # elif args[1] == "P2":
#     #     pregunta_2()
#     # elif args[1] == "P3":
#     #     pregunta_3()
#     # elif args[1] == "P4":
#     #     pregunta_4()
#     # elif args[1] == "P5":
#     #     pregunta_5()
#     # elif args[1] == "P6":
#     #     pregunta_6()
#     # elif args[1] == "ALL":
#     #     pregunta_1()
#     #     pregunta_2()
#     #     pregunta_3()
#     #     pregunta_4()
#     #     pregunta_5()
#     #     pregunta_6()
#     # else:
#     #     print("Wrong usage. Call using one of the following options:")
#     #     for i in range(1, 7):
#     #         print(f" * P{i}")

#     cv2.waitKey(0)


# if __name__ == '__main__':
#     exit(main(sys.argv))
