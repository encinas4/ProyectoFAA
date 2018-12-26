import numpy as np
import matplotlib.pyplot as plt
from generarCeldas import extrae_celdas, parametros_por_defecto

class Letrita:
    celda = None
    clase = -1
    letra = ''
    
    def __init__(self, clase, letra, celda):

        self.clase = clase
        self.letra = letra
        self.celda = celda



def guardaLetritas(celdas, clases):
    letritas =np.array(())
    i = 0
    umbral = 155
    for celda in celdas:
        if i == 10:
            i = 0
        celdaBinario = (celda>umbral)
        letrita = Letrita(i,clases[i], celdaBinario)
        letritas = np.append(letritas, letrita)
        i += 1

    return letritas

##Funcion que recorta una imagen

def recortarImagen(imagen):
    aux = []
    for linea in imagen:
        if False in linea:
            aux.append(linea)

    return np.array(aux)


def recortarCelda(celda):
    celdaRecortada = celda[10:-10,10:-10]
    celdaRecortada = recortarImagen(celdaRecortada)
    celdaRecortada = np.transpose(recortarImagen(np.transpose(celdaRecortada)))

    return celdaRecortada

def recortarLetritas(letritas):

    letritasRecortadas = np.array(())
    for letrita in letritas:
        celdaRecortada = recortarCelda(letrita.celda)
        letritaRecortada = Letrita(letrita.clase, letrita.letra, celdaRecortada)
        letritasRecortadas = np.append(letritasRecortadas, letritaRecortada)

    return letritasRecortadas

