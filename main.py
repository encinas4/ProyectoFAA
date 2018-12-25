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
    datos = generacionDatos(letritas, letritasRecortadas, [True, True, True, True, True, True, True], 20, 20, 15)

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

    #nb = np.array(())
    #knn = np.array(())
    #rl = np.array(())
    #rf = np.array(())
    #rn = np.array(())
    #ad = np.array(())

    #5. clasificamos
    #for i in range(5):
    #    nb = np.append(nb, clasificaNaiveBayes(datos, divisionVC))
    #    knn = np.append(knn, clasificaKNN(datos, 5, divisionVC))
    #    rl = np.append(rl, clasificaRegresionLogistica(datos, 750, divisionVC))
    #    rf = np.append(rf, clasificaRandomForest(datos, 500, 2, 1,divisionVC))
    #    rn = np.append(rn, clasificaRedNeuronal(datos, 750, divisionVC))
    #    ad = np.append(ad, clasificaArbolDecision(datos, divisionVC))

    #print("Media aciertos NB con 5 ejecuciones: ", (np.mean(nb)*100))
    #print("Media aciertos KNN con 5 ejecuciones: ", (np.mean(knn)*100))
    #print("Media aciertos RL con 5 ejecuciones: ", (np.mean(rl)*100))
    #print("Media aciertos RF con 5 ejecuciones: ", (np.mean(rf)*100))
    #print("Media aciertos RN con 5 ejecuciones: ", (np.mean(rn)*100))
    #print("Media aciertos AD con 5 ejecuciones: ", (np.mean(ad)*100))



if __name__ == '__main__':
	main()