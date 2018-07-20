__all__ = (
    'detect',
    'post_process',
)


import collections
import itertools
import math
import sys

import cv2
import numpy as np
import tensorflow as tf


import full_tensor_ia


DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARS = LETTERS + DIGITS

###
# funciones de activacion
def softmax(a):
    exps = np.exp(a.astype(np.float64))
    return exps / np.sum(exps, axis=-1)[:, np.newaxis]

def sigmoid(a):
  return 1. / (1. + np.exp(-a))



def image_coords_size(img,min_shape):
    radio = 1. / 2 ** 0.5
    shape = (img.shape[0] / radio, img.shape[1] / radio)

    while True:
        shape = (int(shape[0] * radio), int(shape[1] * radio))
        if shape[0] < min_shape[0] or shape[1] < min_shape[1]:
            break
        yield cv2.resize(img, (shape[1], shape[0]))


def detect(img, parametros_vals):
   
    # convertir imagen ...

    scaled_ims = list(image_coords_size(img, full_tensor_ia.WINDOW_SHAPE))

    # cargar el modelo para detectar los caracteres

    x, y, params = full_tensor_ia.get_detect_model()

    # ejecutar el modelo para cada escala

    with tf.Session(config=tf.ConfigProto()) as sess:
        y_vals = []
        for scaled_im in scaled_ims:
            feed_dict = {x: np.stack([scaled_im])}
            feed_dict.update(dict(zip(params, parametros_vals)))
            y_vals.append(sess.run(y, feed_dict=feed_dict))

    #retorna las coordenadas de la caja donde reconocemos la placa
    # , la matriz de probabilidades de los castacteres

    # chacarcater prob , es la probabilidad de los caracreres 
    for i, (scaled_im, y_val) in enumerate(zip(scaled_ims, y_vals)):
        for window_coords in np.argwhere(y_val[0, :, :, 0] >
                                                       -math.log(1./0.99 - 1)):
            character_prob = (y_val[0,
                                  window_coords[0],
                                  window_coords[1], 1:].reshape(
                                    7, len(CHARS)))
            character_prob = softmax(character_prob)

            img_scale = float(img.shape[0]) / scaled_im.shape[0]

            bbox_ini = window_coords * (8, 4) * img_scale
            bbox_size_fin = np.array(full_tensor_ia.WINDOW_SHAPE) * img_scale

            present_prob = sigmoid(
                               y_val[0, window_coords[0], window_coords[1], 0])

            yield bbox_ini, bbox_ini + bbox_size_fin, present_prob, character_prob


def _overlaps(match1, match2):
    bbox_tl1, bbox_br1, _, _ = match1
    bbox_tl2, bbox_br2, _, _ = match2
    return (bbox_br1[0] > bbox_tl2[0] and
            bbox_br2[0] > bbox_tl1[0] and
            bbox_br1[1] > bbox_tl2[1] and
            bbox_br2[1] > bbox_tl1[1])


def grupos_overlapping_rect(matches):
    matches = list(matches)
    num_grupos = 0
    match_to_grupo = {}
    for idx1 in range(len(matches)):
        for idx2 in range(idx1):
            if _overlaps(matches[idx1], matches[idx2]):
                match_to_grupo[idx1] = match_to_grupo[idx2]
                break
        else:
            match_to_grupo[idx1] = num_grupos 
            num_grupos += 1

    grupos = collections.defaultdict(list)
    for idx, group in match_to_grupo.items():
        grupos[group].append(matches[idx])

    return grupos


def post_process(matches):
   
    grupos = grupos_overlapping_rect(matches)

    for group_matches in grupos.values():
        mins = np.stack(np.array(m[0]) for m in group_matches)
        maxs = np.stack(np.array(m[1]) for m in group_matches)
        present_probs = np.array([m[2] for m in group_matches])
        character_prob = np.stack(m[3] for m in group_matches)

        yield (np.max(mins, axis=0).flatten(),
               np.min(maxs, axis=0).flatten(),
               np.max(present_probs),
               character_prob[np.argmax(present_probs)])


def characters_get_code(character_prob):
    return "".join(CHARS[i] for i in np.argmax(character_prob, axis=1))


if __name__ == "__main__":
    for i in range(1,11):
        imagen=str(i)+'.jpg'
        
        print "IMAGEN",imagen
        #imagen=''

        salida=str(i)+'_salida.jpg'
        img = cv2.imread(imagen)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.

        f = np.load(sys.argv[1])
        parametros_vals = [f[n] for n in sorted(f.files, key=lambda s: int(s[4:]))]

        for pt1, pt2, present_prob, character_prob in post_process(
                                                      detect(img_gray, parametros_vals)):
            pt1 = tuple(reversed(map(int, pt1)))
            pt2 = tuple(reversed(map(int, pt2)))

            code = characters_get_code(character_prob)

            color = (255.0, 0.0, 255.0)
            dic="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            print pt1
            print pt2
            x=[]
            x=character_prob.tolist()
            #print x
            for i in range(len(x)):
                for j in range(len(x[0])):
                    if x[i][j]>0.2:
                        print "Caracter: "
                        print dic[j]
                        print "Porcentaje"
                        print x[i][j]*100,'%'
                print "____________________"

        
            print "Placa: ",code    
            cv2.rectangle(img, pt1, pt2, color)

           

        cv2.imwrite(salida, img)
        imagen=''
        salida=''
        
