---
title: Tarea 1 - Procesamiento Digital de imágenes
author:
  - name: Pablo Yáñez S.
    email: pablo.yanez@uai.cl
numbersections: yes
lang: es
# abstract: El siguiento documento presenta el desarrollo de las actividades descritas en el enunciado de la tarea 1 del ramo.
header-includes: |
  \usepackage{booktabs}
  \usepackage{flushend}

---

# Selección de imagen

Para el desarrollo se utiliza un retrato encontrado en el sitio [www.pexels.com](pexels.com).
La fotografía presentanda en @fig:retrato. fue tomada por Andrea Piacquadio ([\@andreapiacquadio_](https://www.instagram.com/andreapiacquadio_/)).

![[Retrato a usar.](https://www.pexels.com/photo/women-s-white-and-black-button-up-collared-shirt-774909/)](portrait.jpg){#fig:retrato}

Al estudiar los distintos canales de la imagen original se puede apreciar que los canales azul y verde se parecen bastante, mientras que el canal rojo se ve mucho más distinto que los otro dos. En especial en este último se aprecian más pixeles con intensidades, especificamente en la zona del rostro de la persona.

![Canales del retrato seleccionado.](outs/bgr.jpg){#fig:canales_retrato}

# Equalización de la imagen

La ecualización de los histogramas de la imagen corresponde al proceso de modificar los valores de intesidad de la imagen, de modo que los valores se encuentren en todo el rango. Esto permite que en la zonas de bajo constraste este aumente.

![Ecualización de los canalaes de la imagen.](outs/bgr_eq.jpg){#fig:canales_eq}

En el resulado presentado en @fig:canales_eq se puede apreciar que el canal que se mas afectado se vio por la ecualización es el canal rojo.

# Correción gamma

Se aplica corrección gamma a cada uno del los canales de forma independiente. Esto consiste en transformar cada uno de los valores de cada unos de los pixeles de acuerdo a la siguiente función:

$$S = r^{\gamma}$$

Valores de $\gamma < 1$ tiene el efecto de aumentar la intensidad de los pixeles de la imagen, produciendo el efecto de "aclarar" la imagen. Mientra que elegir valores $\gamma > 1$ disminuyen la intensidad de los pixeles, produciendo el efecto de "oscurece" la imagen.

Dado que a priori se desconce el valor de $\gamma$ que se desea aplicar a cada canal, se aplican valores entre $[0{.}5 - 1{.}9]$. A modo de ejemplo en @fig:gamma_red se muestra la imagen generada para el canal rojo.

![Ejemplo variacion parámetro en correción gamma para canal rojo.](outs/gamma_R_simple.jpg){#fig:gamma_red}

Luego de una inspección de los resultados para los distintos valores de $\gamma$ para cada uno de los canales se opta por elegir los valores de $\gamma_b=0{.}8$, $\gamma_g=0{.}9$ y $\gamma_r=1{.}2$ para cada uno de los canales. El resultado obtenido para cada canal se presenta en el @fig:canales_gamma.

![Correción gamma por canal.](outs/gamma.jpg){#fig:canales_gamma}


# Filtro de la mediana

El filtro de la media es un filtro que se utiliza comúnmente para reducir el ruido de una imagen. En este se reemplaza el valor de un pixel en relación a sus vecinos. Los vecinos y el pixel se ordenan, y el valor del pixel se reemplaza por el valor de la mediana del os datos. La elección de la cantidad de vecinos se define en lo que se conoce como máscara.

![Filtro de la mediana por canal (3x3).](outs/median_3.jpg){#fig:median_3}

En @fig:median_3 se presenta el resultado de aplicar un filtro de la mediana con tamaño de máscara de 3x3. En comparación a la imagen original, el resultado se ve como si estuviese suavizado, perdiendo definición en los detalles. En especial se observa que se pierden algunas de las imperfecciones presentes en el rostro.

![Filtro de la mediana por canal (5x5).](outs/median_5.jpg){#fig:median_5}

Al agrandar la máscara del filtro se acentúa la difuminación de la imagen. En @fig:median_5 se puede nota se nota como se empiezan a deformar las figuras geometricas de la blusa de la persona.

# Imagen final

Utilizando las imágenes previamente generadas. En @fig:to_merge se presentan las versiones a color de cada una de las imágenes previamente generadas.

![imágenes a combinar.](outs/color_merge.jpg){#fig:to_merge}

Las imágenes se suman de forma ponderada para generar una imagen única. Se considera la siguiente ponderación para cada una de las imagenes:

- Imagen equalizada: $35%$
- Imagen correcion gamma: $35%$
- Filtro de la mediana 3x3: $15%$
- Filtro de la mediana 5x5: $15%$

![imagen original vs final.](outs/before_after.jpg){#fig:final}

En @fig:final se puede comparar el antes y después de la imagen una vez que se han realizado todas las manipulaciones. Se puede apreciar que cambia drasticamente la tonalidad de la imagen, pasando de una tonalidad calida a una mas fría, cambios que se pueden atribuir a las manipulaciones realizadas a través de la equalización y la correción gamma. Las imperfecciones del rostro, especificamente en la zona de la frente, se pierden gracias al efecto de los filtros de la mediana.


# Manual de uso

En conjunto a este documento se entrega el código fuente asociado al desarrollo de este trabajo.

Para recrear los resultados obtenidos en este documento basta con ejecutar el script `Tarea_1.py` agregando como opción la pregunta asociada.

```
# Ejecuta pregunta 1
Tarea_1.py P1

# Ejecuta todo
Tarea_1.py ALL
```


# References

---
nocite: '@*'
---

