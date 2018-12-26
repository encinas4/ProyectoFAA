import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from generarCeldas import detectar_lineas, extrae_cuadradito, extrae_celdas, parametros_por_defecto, guarda_celdas, mostrar_imagenes
from letrita import guardaLetritas, recortarCelda, Letrita
from generarDatos import generacionDatos

def test_celdas():
    """ Recorre todas las imágenes, extrae las celdas y guarda
        el resultado en ficheros de imagen individuales
    """
    
    nombres_imagenes, \
    alto,             \
    ancho,            \
    clases           = parametros_por_defecto()
    
    celdas = extrae_celdas(nombres_imagenes, alto, ancho)
    
    guarda_celdas(celdas, clases, 'out')

    mostrar_imagenes(celdas[:110])

    return celdas, clases


def test_imagen():
    imagen = mpimg.imread('datos/out-0032.png', True)
    h, v = detectar_lineas(imagen)
    aux = extrae_cuadradito(imagen,h[np.random.randint(2,13)],v[np.random.randint(2,11)],210,150)
    plt.imshow(aux,cmap='gray')


def test_letritas(celdas, clases):

    letritas = guardaLetritas(celdas, clases)
    print("Ejemplo de letritas guardadas")
    plt.figure(figsize=(10,10))
    for i,iimg in enumerate(np.random.randint(0, len(letritas),size=100)):
        plt.subplot(10,10,i+1)
        plt.imshow(letritas[iimg].celda)

    return letritas

def test_letritas_recortadas(letritas):

    letritasRecortadas = np.array(())
    for letrita in letritas:
        celdaRecortada = recortarCelda(letrita.celda)
        letritaRecortada = Letrita(letrita.clase, letrita.letra, celdaRecortada)
        letritasRecortadas = np.append(letritasRecortadas, letritaRecortada)
    
    print("Ejemplo de letritas recortadas")
    plt.figure(figsize=(10,10))
    for i,iimg in enumerate(np.random.randint(0, len(letritasRecortadas),size=100)):
        plt.subplot(10,10,i+1)
        plt.imshow(letritasRecortadas[iimg].celda)

    return letritasRecortadas

def test_datos(letritas, letritasRecortadas):
    datos = datos = generacionDatos(letritas, letritasRecortadas, [True, True, True, True, True, True, True], 5, 5, 3)

    print("Matriz de datos obtenida: ")
    print(datos)
