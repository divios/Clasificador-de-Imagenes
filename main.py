#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                       %      
% TDImagen PROYECTO 2: CLASIFICACIÓN DE IMÁGENES                        %
%                                                                       %
% Plantilla para implementar el sistema de clasificación de imágenes    %
% del proyecto, que permitirá determinar si una habitación está         %
% ordenada o desordenada.                                               %       
%                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
import os, natsort
import numpy as np
from skimage import io, filters, color, util
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
from skimage import exposure
from scipy.stats import entropy
from skimage.feature import canny, greycomatrix, greycoprops
import matplotlib.pyplot as plt

def imageProcessing(image):
  """ La función recibe como entrada la imagen 'image' a
      procesar. A la salida devuelve una variable de tipo
      diccionario ('processed_images') que contiene indexadas
      cada una de las imágenes resultantes de las diferentes
      técnicas de procesado escogidas (imagen filtrada,
      imagen de bordes, máscara de segmentación, etc.), utilizando para cada
      una de ellas una key que identifique su contenido 
      (ej. 'image', 'mask', 'edges', etc.).
      
      Para más información sobre las variables de tipo 
      diccionario en Python, puede consultar el siguiente enlace:
       https://www.w3schools.com/python/python_dictionaries.asp 
      
      COMPLETE la función para obtener el conjunto de imágenes procesadas
      'processed_images' necesarias para la posterior extracción de características.
      Si lo necesita, puede definir y hacer uso de funciones auxiliares.
  """
  # El siguiente código implementa el BASELINE incluido en el challenge de
  # Kaggle. 
  # - - - MODIFICAR PARA IMPLEMENTACIÓN DE LA SOLUCIÓN PROPUESTA. - - -
  processed_images = {}
  # Ejemplo: Añadimos la imagen original como una entrada a la variable diccionario
  processed_images["image"] = image
  # Añadimos la imagen en escala de grises como una entrada a la variable diccionario
  img_gray = color.rgb2gray(image)
  processed_images["image_gray"] = img_gray
  #Calculamos histograma normalizado de la imagen en gris y lo añadimos
  gray_hist,_ = exposure.histogram(img_gray,nbins = 32, normalize=True)
  processed_images["hist_norm"] = gray_hist
  #Calculamos bordes de la imagen
  img_edges = canny(img_gray, high_threshold= 0.3)
  processed_images["edges"] = img_edges
  #Calculamos la transformada de fourier
  
  #Calculamos la matriz de co...
  aux = util.img_as_ubyte(img_gray)
  properties = ['correlation']

  glcm = greycomatrix(aux, 
                  distances=[1], 
                  angles=[0],
                  symmetric=True,
                  normed=True)

  processed_images["comatrix"] = np.hstack([greycoprops(glcm, prop).ravel() for prop in properties])
  
  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  
  return processed_images

def extractFeatures(processed_images):  
  """ La función recibe como entrada la variable 'processed_images'
      de tipo diccionario, la cual contiene las imágenes procesadas
      obtenidas a partir de cada imagen de la base de datos, necesarias
      para la extracción de las características de dicha imagen. A la salida
      devuelve un vector con los valores de descriptores obtenidos 
      para la imagen.
      
      COMPLETE la función para obtener el vector de características
      escogidas 'features' para representar una imagen de la categoría 
      que se le ha asignado. Si lo necesita, puede definir y hacer uso de 
      funciones auxiliares.
  """
  # El siguiente código implementa el BASELINE incluido en el challenge de
  # Kaggle. 
  # - - - MODIFICAR PARA IMPLEMENTACIÓN DE LA SOLUCIÓN PROPUESTA. - - -
  features = []
  # Utilizamos la imagen en escala de grises para obtener, como descriptor
  # baseline, su desviación típica (descriptor de textura).
  image_gray = processed_images["image_gray"]
  
  hist_mean = np.mean(image_gray)
  features.append(hist_mean)
  
  std_gray = np.std(image_gray)
  #features.append(std_gray)
  
  gray_hist = processed_images["hist_norm"]
  entropia = entropy(gray_hist)
  #features.append(entropia)
  
  img_edges = processed_images["edges"]
  n_edges = np.count_nonzero(img_edges)
  features.append(n_edges)
  
  f = np.fft.fft2(image_gray)
  fshift = np.fft.fftshift(f)
  magnitude_spectrum = 20*np.log(np.abs(fshift))
  fourier_mean = np.mean(magnitude_spectrum)
  fourier_std = np.std(magnitude_spectrum)
  
  features.append(fourier_std)
  #features.append(fourier_mean)
  
  comatrix = processed_images["comatrix"]
  features = np.concatenate((features, comatrix))
  
 
  
  # Concatenamos todas las features obtenidas (si son más de 1)
  # features = np.concatenate(features)
  
  
  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  return features
  
def databaseFeatures(database_set_root="train"):
  """ La función recibe como entrada una variable:
      - 'database_set_root', que es la ruta a la carpeta que contiene las
      imágenes de uno de los dos conjuntos de la base de datos: "train" o "test"
      
      A su salida, la función devuelve la matriz de características 'X' y 
      que representa dicho conjunto de la base de datos. 
      Estas variable se utilizará en la etapa de clasificación del sistema.
      
      ÚNICAMENTE ES NECESARIO MODIFICAR LA VARIABLE 'num_features' EN
      ESTA FUNCIÓN.
  """
  image_roots = natsort.natsorted([name for name in os.listdir(database_set_root) if name.endswith('.png')])
  
  # Matriz de caracteristicas X
  # Para el BASELINE incluido en el challenge de Kaggle, se utiliza 1 feature
  num_features = 4 #MODIFICAR, INDICANDO EL NÚMERO DE CARACTERÍSTICAS EXTRAÍDAS
  X=np.zeros((np.size(image_roots),num_features))
  for i in np.arange(np.size(image_roots)):
      print(database_set_root+'/'+image_roots[i])
      
      # Cargar imagen
      image = io.imread(database_set_root+'/'+image_roots[i])
      
      # PREPROCESADO (ver función imageProcessing)
      processed_images = imageProcessing(image)
      
      # EXTRACCION DE CARACTERISTICAS (ver función extractFeatures)
      X[i, :]
  = extractFeatures(processed_images)
  
  return X

def train_classifier(X_train, y_train, X_val = [], y_val = []):
  """ La función recibe como entrada:
  
      - Las variables 'X_train', 'y_train', matriz de características
      y vector de etiquetas del conjunto de entrenamiento, respectivamente.
      Permiten obtenener un modelo de clasificación, a escoger por el equipo.
      
      El vector de etiquetas y es binario: asigna un 0 a imágenes
      de habitaciones ordenadas y un 0 a imágenes de habitaciones
      desordenadas. De esta manera, permite resolver un problema de clasificación
      binaria en el que se discrimine entre las dos categorías.
      
      - Las variables 'X_val', 'y_val', matriz de características y 
      vector de etiquetas del conjunto de validación, respectivamente.
      Permiten validar los parámetros del algoritmo de clasificación escogido.
      Son variables opcionales.
      
      NOTA: Si se desea realizar un procedimiento de validación cruzada del
      modelo, puede subdividir el conjunto de entrenamiento ('X_train','y_train')
      en las particiones necesarias dentro de esta función.
      
      A su salida, la función devuelve la variable 'scaler', para normalizar
      los datos de entrada, así como el modelo obtenido a partir
      del conjunto de imágenes de entrenamiento, 'model'.
      
      COMPLETE la función para obtener un modelo de clasificación binaria
      que permita discriminar su categoría de las asignadas al resto de equipos.
      
  """
  # El siguiente código implementa el clasificador utilizado para el BASELINE 
  # incluido en el challenge de Kaggle. 
  # - - - NO ES NECESARIO MODIFICAR EL CLASIFICADOR (SÍ SUS PARÁMETROS) PERO, 
  # SI LO DESEA, PUEDE HACERLO Y SU PROPUESTA SE VALORARÁ POSITIVAMENTE - - - 
  
  # Normalización ((X-media)/std >> media=0, std=1)
  scaler = StandardScaler()
  # Obtener estadísticos del conjunto de entrenamiento 
  scaler.fit(X_train)
  # Normalización del conjunto de entrenamiento, aplicando los estadísticos.
  X_train = scaler.transform(X_train)

  # Definición y entrenamiento del modelo
  model = MLPClassifier(hidden_layer_sizes=(np.maximum(10,np.ceil(np.shape(X_train)[1]/2).astype('uint8')),
                                            np.maximum(5,np.ceil(np.shape(X_train)[1]/4).astype('uint8'))),
                        max_iter=200, alpha=1e-4, solver='sgd', verbose=10, random_state=1,
                        learning_rate_init=.1)
  model.fit(X_train, y_train)
  ### - - - - - - - - - - - - - - - - - - - - - - - - -
  
  return scaler, model

def test_classifier(scaler, model, X_test):
  """ La función recibe como entrada:
  
      - La variable 'scaler', que permite normalizar las características de
      las imágenes del conjunto de test de acuerdo con los estadísticos obtenidos
      a partir del conjunto de entrenamiento.
      - La variable 'model', que contiene el modelo de clasificación binaria obtenido
      mediante la función 'train_classifier' superior.
      - La variable 'X_test', matriz de características del conjunto de test, sobre
      la que se evaluará el modelo de clasificación obtenido a partir del conjunto
      de entrenamiento.
  
      A su salida, la función devuelve el vector de etiquetas predichas
      para las imágenes del conjunto de test. 
      
      COMPLETE la función para obtener el vector de etiquetas predichas 
      'y_pred' para el conjunto 'X_test'.
  """
  # El siguiente código implementa el clasificador utilizado para el BASELINE 
  # incluido en el challenge de Kaggle. 
  # - - - NO ES NECESARIO MODIFICAR EL CLASIFICADOR (SÍ SUS PARÁMETROS) PERO, 
  # SI LO DESEA, PUEDE HACERLO Y SU PROPUESTA SE VALORARÁ POSITIVAMENTE - - - 
  
  # Normalización del conjunto de test, aplicando los estadísticos del 
  # conjunto de entrenamiento
  X_test = scaler.transform(X_test)

  # Prediccciones probabilísticas sobre el conjunto de test
  y_pred = model.predict_proba(X_test)
  y_pred = y_pred[:,1] # Score correspondiente a la clase 1 (positiva)
  ### - - - - - - - - - - - - - - - - - - - - - - - - -
  
  return y_pred 

def eval_classifier(y, y_pred):
  """ La función recibe a su entrada el vector de etiquetas
      reales (Ground-Truth (GT)) 'y', así como el vector
      de etiquetas predichas 'y_pred' del conjunto de imágenes
      que se utiliza para evaluar las prestaciones del 
      clasificador.
      
      A su salida, devuelve la variable 'score', con el valor
      obtenido para la medida de evaluación considerada 
      ('AUC' en este proyecto).
      
      Esta función se puede utilizar para realizar pruebas
      en local, siempre sobre un subconjunto de validación
      del conjunto de entrenamiento, del cual se dispongan 
      las etiquetas.
  """
  fpr, tpr, thresholds = roc_curve(y, y_pred, pos_label=1)
  auc_score = auc(fpr, tpr)
  print('AUC sobre el conjunto de imágenes proporcionado: '+str(auc_score))
  return auc_score
