from clasificadores import clasificaNaiveBayes, clasificaKNN, clasificaRegresionLogistica, clasificaRandomForest
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
    print("Media de aciertos de KNN con k = 3")
    print(clasificaKNN(datos, 15, divisionVC))
    print("Media de aciertos de Regresión Logística con nEpocas = 100")
    print(clasificaRegresionLogistica(datos, 100, divisionVC))
    print("Media de aciertos de RandomForest")
    print(clasificaRandomForest(datos, 200, 2, divisionVC))


if __name__ == '__main__':
	main()