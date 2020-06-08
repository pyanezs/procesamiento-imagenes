import numpy as np


def filtro_mediana_adaptiva(img, s_max=16, ws=4):

    img_out = np.zeros(img.shape)
    m, n = img.shape
    coords = [(i, j) for i in range(m - s_max) for j in range(n - s_max)]

    for coord in coords:
        i, j = coord
        zxy = img[i, j]
        wadapt = ws

        while True:
            roi = img[i:(i + wadapt), j:(j + wadapt)]

            size = wadapt <= s_max
            A = condition_A(roi)
            B = condition_B(roi, zxy)

            if size and A and B:
                img_out[i, j] = zxy
                break
            elif size and A and not B:
                img_out[i, j] = np.median(roi.flatten())
                break
            elif size and not A:
                wadapt += 1
                break
            else:
                img_out[i, j] = zxy
                break

    return (np.uint8(img_out[0:(m - s_max), 0:(n-s_max)]))


def condition_A(img):

    z_min = np.min(img.flatten())
    z_max = np.max(img.flatten())
    z_med = np.median(img.flatten())

    a1 = z_med - z_min
    a2 = z_med - z_max

    return (a1 > 0) & (a2 < 0)


def condition_B(img, zxy):

    z_min = np.min(img.flatten())
    z_max = np.max(img.flatten())

    b1 = float(zxy)-float(z_min)
    b2 = float(zxy)-float(z_max)

    return (b1 > 0) & (b2 < 0)



# Prueba Filtro

