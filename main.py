from clasificadores import clasificaNaiveBayes, clasificaKNN, clasificaRegresionLogistica, clasificaRandomForest, clasificaRedNeuronal, clasificaArbolDecision
from generarDatos import generacionDatos
from letrita import guardaLetritas, recortarLetritas
from generarCeldas import extrae_celdas, parametros_por_defecto
import numpy as np

def main():

    divisionVC = 10
    #1. conseguimos las celdas
    nombres_imagenes, \
    alto,             \
    ancho,            \
    clases           = parametros_por_defecto()

    celdas = extrae_celdas(nombres_imagenes, alto, ancho)
    
    # 2. creamos las letritas
    letritas = guardaLetritas(celdas, clases)

    #3. recortamos las letritas
    letritasRecortadas = recortarLetritas(letritas)

    #4 . creamos la matriz de datos (segun queramos)
    datos = generacionDatos(letritas, letritasRecortadas, [True, True, True, True, True, True, True], 17, 17, 6)

    #5. clasificamos
    print("Media de aciertos de Naive Bayes")
    print(clasificaNaiveBayes(datos, divisionVC))
    print("Media de aciertos de KNN con k = 5")
    print(clasificaKNN(datos, 5, divisionVC))
    print("Media de aciertos de Regresión Logística con nEpocas = 1000")
    print(clasificaRegresionLogistica(datos, 750, divisionVC))
    print("Media de aciertos de RandomForest")
    print(clasificaRandomForest(datos, 500, 2, 1,divisionVC))
    print("Media de aciertos de Red Neuronal")
    print(clasificaRedNeuronal(datos, 750, divisionVC))
    print("Media de aciertos de Arbol de decision")
    print(clasificaArbolDecision(datos, divisionVC))
    #print("Media de aciertos de Red Neuronal")
    #print(clasificaRedNeuronalNuevo(datos, 750, divisionVC))


if __name__ == '__main__':
	main()