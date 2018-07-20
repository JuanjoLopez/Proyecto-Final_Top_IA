# Reconocimiento de placas de autos utilizando Redes Convolucionales(CNN)
> Este proyecto consiste en identificar la placas de los autos y además en reconocer los carácteres que estan presente en ella,utilizando 
redes convolucionales,fue realizado para la presentación de trabajo final del curso Tópicos en Inteligencia Artificial-UNSA



## Integrantes
* **Oliver Sumari Huayta**
* **Juan López Condori**
* **Eyner Pariguana Medina**


## Introducción
> Para este trabajo hemos estado interesados en deep learning, en particular las redes neuronales convolucionales. Un documento sobresaliente de los últimos tiempos es el reconocimiento de números (de múltiples dígitos). Esta investigación describe un sistema para extraer números de casas de las imágenes de Google Street View utilizando una única red neuronal. Luego, los autores explican cómo se puede aplicar la misma red para romper el propio sistema CAPTCHA de Google con precisión a nivel humano.

> Con el fin de obtener experiencia práctica con la implementación de redes neuronales, decidimos diseñar un sistema para resolver un problema similar: reconocimiento automático de placas de automóviles. Para este proyecto hemos usado Python, TensorFlow, OpenCV y NumPy.

## Data

> Para simplificar el procesamiento de imágenes para el entrenamiento, decidimos que la red funcionase en imágenes de escala de grises de 128x64. Se eligió esa dimensión como resolución de entrada, ya que es lo suficientemente pequeña como para permitir el entrenamiento en un tiempo razonable con recursos modestos, pero también lo suficientemente grande como para que las placas sean legibles.

![](https://matthewearl.github.io/assets/cnn-anpr/window-example.jpg)

> Para cada frame que la red devuelva, entregará dos cosas. La primera es la probabiliad que el número de la placa es presentado en la imagen de entrada. Y la segunda es la probabilidad de cada dígito en cada posición. Por ejemplo, en cada una de las 7 posibles posiciones debería retornar una probabilidad entre los 36 caracteres existentes (número, letras). Para este proyecto asumimos un tipo de secuencia, debido que no en todos los paises es la misma secuencia de caracteres, además que con esta secuencia ya hemos entrenado nuestro proyecto. Esta secuencia esta dada por 7 carcateres (Letra - Letra - Número - Número - Letra - Letra - Letra).

> Se cosidera como placa si está dentro de los bordes de la imagen. Si el ancho es menor que el 80% de la imagen y mayor que el 60% y la altura es menor al 87.5% de de la altura de la imagen y mayor al 60%.  


## Arquitectura Utilizada
> La arquitectura utilizada fue la siguiente:

![](https://matthewearl.github.io/assets/cnn-anpr/topology.svg)

## Implementado con
* Python-Lenguaje de Programación usado.
* [TensorFlow](https://www.tensorflow.org/) 
* [OpenCV](http://opencv.org/) 
* [Numpy](http://www.numpy.org/) 


## Aplicaciones 

  * En video vigilancia.
  * Para el seguimientos de autos que infringen una norma de tráfico.
  * Reconocer autos implicados en delitos.
  * Para sistemas de parqueos inteligentes.

## Referencia:
Paper: http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42241.pdf
