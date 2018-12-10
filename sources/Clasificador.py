from abc import ABCMeta, abstractmethod
from scipy.spatial import distance
import numpy as np
import math
import random
import decimal


class matrizConfusion:

    VP = None
    VN = None
    FP = None
    FN = None

    def __init__(self):
            self.VP = -1
            self.VN = -1
            self.FP = -1
            self.FN = -1


class Clasificador:
    # Clase abstracta
    __metaclass__ = ABCMeta

    numErrores = 0
    numAciertos = 0
    nombreClasificador = "null"
    matrizConf = None

    # Metodos abstractos que se implementan en casa clasificador concreto
    @abstractmethod
    # TODO: esta funcion deben ser implementadas en cada clasificador concreto
    # datosTrain: matriz numpy con los datos de entrenamiento
    # atributosDiscretos: array bool con la indicatriz de los atributos nominales
    # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion
    # de variables discretas
    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario):
        pass

    @abstractmethod
    # TODO: esta funcion deben ser implementadas en cada clasificador concreto
    # devuelve un numpy array con las predicciones
    def clasifica(self, datosTest, atributosDiscretos, diccionario):
        pass

    def media(self, datos):

        media = 0
        for dato in datos:
            media += dato
        media /= float(len(datos))

        return media

    def desviacionTipica(self, datos):
        if len(datos) == 1:

            desviacionTipica = 0.0

        else: 

            media = self.media(datos)
            varianza = 0
            for dato in datos:
                varianza += pow(dato - media, 2)
            varianza /= float(len(datos) - 1)
            desviacionTipica = math.sqrt(varianza)

        return desviacionTipica

    # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
    # TODO: implementar
    def error(self, datos, pred):
        aciertos = 0
        falsosPositivos = 0
        verdaderoNegativo = 0
        verdaderoPositivo = 0
        falsosNegativos = 0
        errores = 0
        # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error
        for i in range(len(pred)):
            if(datos[i][-1] == pred[i]):
                if (datos[i][-1] == 0) and (pred[i] == 0):
                    verdaderoNegativo += 1

                elif (datos[i][-1] == 1) and (pred[i] == 1):
                    verdaderoPositivo += 1
                aciertos += 1
            else:
                if (datos[i][-1] == 0) and (pred[i] == 1):
                    falsosPositivos += 1

                elif (datos[i][-1] == 1) and (pred[i] == 0):
                    falsosNegativos += 1
                errores += 1
        self.matrizConf = matrizConfusion()
        self.matrizConf.VP = verdaderoPositivo
        self.matrizConf.VN = verdaderoNegativo
        self.matrizConf.FP = falsosPositivos
        self.matrizConf.FN = falsosNegativos
        return errores/len(pred)

    # Realiza una clasificacion utilizando una estrategia de particionado determinada
    # TODO: implementar esta funcion
    def validacion(self, particionado, dataset, clasificador, seed=None):
        # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
        # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
        # y obtenemos el error en la particion de test i
        # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
        # y obtenemos el error en la particion test
        errores = []
        particiones = []
        particiones = particionado.creaParticiones(dataset)
        datosTrain = []
        datosTest = []

        for particion in particiones:
            #NUEVO, es para sacar una lista de los datos, ya que ahora tenemos indices de datos y tenemos que sacar los datos propios
            for indiceTrain in particion.indicesTrain:
                datosTrain.append(dataset.extraeDatos(indiceTrain))
            for indiceTest in particion.indicesTest:
                datosTest.append(dataset.extraeDatos(indiceTest))
            clasificador.entrenamiento(datosTrain, dataset.nominalAtributos, dataset.diccionarios)
            predicciones = clasificador.clasifica(datosTest, dataset.nominalAtributos, dataset.diccionarios)
            error = self.error(datosTest, predicciones)
            errores.append(error)
            clasificador.reset()

        mediaError = self.media(errores)
        desviacionError = self.desviacionTipica(errores)
        print('Media de error: ' + str(mediaError))
        print('Desviacion tipica de error: ' + str(desviacionError))
        pass


##############################################################################

class ClasificadorNaiveBayes(Clasificador):
    mediasEntrenamiento = []
    desviacionesEntrenamiento = []
    resultadosEntrenamiento = {}
    resultadosDatosDiscretos = []
    numClases = 0
    totalClase = []
    priores = []
    laplace = True

    def __init__(self):
        self.nombreClasificador = "NaiveBayes"

    def reset(self):
        self.priores = []
        self.numClases = 0
        self.totalClase = 0
        self.mediasEntrenamiento = []
        self.desviacionesEntrenamiento = []
        self.resultadosEntrenamiento = {}
        self.resultadosDatosDiscretos = []
        datosTrain = []
        datosTest = []
        self.laplace = True

    def procesaClase(self, datosClase, atributosDiscretos):
        resultadosClase = []
        i = 0
        valoresAtributos = None
        # Generamos una lista de longitud numero de atributos que contiene listas con los valores de cada atributo
        valoresAtributos = zip(*datosClase)
        # Calculamos media y desviacion tipica y lo guardamos en la tabla de resultados
        for atributo in valoresAtributos:
            if(atributosDiscretos[i] == False):
                resultadosClase.append(
                    (self.media(atributo), self.desviacionTipica(atributo)))
            else:
                resultadosClase.append([])
            i += 1
        # Borramos el ultimo porque contiene la clase
        del resultadosClase[-1]
        return resultadosClase

    def aplicaLaplace(self, tabla):
        # Sumamos 1 ca cada celda de la tabla y al total de elementos de la clase de cada celda
        for valorAtributo in tabla.keys():
            for i in range(len(tabla[valorAtributo])):
                tabla[valorAtributo][i] += 1
                self.totalClase[i] += 1
        return tabla

    # TODO: implementar
    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):
        datosClases = {}

        # Cogemos los datos por clases el ultimo elemento de cada dato contiene la clase
        for i in range(len(datostrain)):
            # Guardamos tabla atributos continuos
            if(datostrain[i][-1] not in datosClases):
                datosClases[datostrain[i][-1]] = []
                self.numClases += 1
            datosClases[datostrain[i][-1]].append(datostrain[i])

        # Calculamos la tabla de entrenamiento de atributos continuos
        for clase, datos in datosClases.items():
            self.resultadosEntrenamiento[int(clase)] = self.procesaClase(
                datos, atributosDiscretos)

        # Generamos tablas para los atributos discretos
        self.resultadosDatosDiscretos = [{}
                                         for _ in range(len(atributosDiscretos)-1)]
        for i in range(len(atributosDiscretos) - 1):
            if(atributosDiscretos[i] == True):
                for j in range(len(diccionario[i])):
                    self.resultadosDatosDiscretos[i][j] = [0] * self.numClases

        self.totalClase = [0] * self.numClases
        # Rellenamos las tablas de los atributos discretos
        for i in range(len(datostrain)):
            # Acumulamos el numero total de elementos de cada clase
            self.totalClase[int(datostrain[i][-1])] += 1
            # Atributos discretos
            for j in range(len(datostrain[i])-1):
                if(atributosDiscretos[j] == True):
                    # Si es discreto rellenamos su tabla
                    self.resultadosDatosDiscretos[j][int(
                        datostrain[i][j])][int(datostrain[i][-1])] += 1

        # Aplicamos Laplace si se indica
        if(self.laplace == True):
            for tabla in self.resultadosDatosDiscretos:
                for valorAtributo in tabla.keys():
                    if(0 in tabla[valorAtributo]):
                        tabla = self.aplicaLaplace(tabla)
                        break

        # Calculamos priores
        self.priores = [0] * self.numClases
        self.totalTrain = sum(self.totalClase)
        for i in range(len(self.totalClase)):
            self.priores[i] = self.totalClase[i] / self.totalTrain

    def probabilidadAtributoDiscretoClase(self, valor, numAtributo, clase):
        return self.resultadosDatosDiscretos[numAtributo][valor][clase] / self.totalClase[clase]

    def probabilidadAtributoContinuoClase(self, valor, media, desviacionTipica):
        exponent = math.exp(-(math.pow(valor-media, 2) /
                              (2*math.pow(desviacionTipica, 2))))
        return (1 / (math.sqrt(2*math.pi) * desviacionTipica)) * exponent

    def probabilidadClases(self, dato, atributosDiscretos):
        probabilidadesClase = {}
        probAtributoClase = 0
        for clase, valoresAtributos in self.resultadosEntrenamiento.items():
            probabilidadesClase[clase] = 1
            for i in range(len(valoresAtributos)):
                if(atributosDiscretos[i] == True):
                    # Atributo discreto
                    probAtributoClase = self.probabilidadAtributoDiscretoClase(
                        dato[i], i, clase)
                else:
                    # Atributo continuo
                    media, desviacion = valoresAtributos[i]
                    valor = dato[i]
                    probAtributoClase = self.probabilidadAtributoContinuoClase(
                        valor, media, desviacion)
                probabilidadesClase[clase] *= probAtributoClase
            probabilidadesClase[clase] *= self.priores[clase]
        return probabilidadesClase

    
    def clasifica(self, datostest, atributosDiscretos, diccionario):
        predicciones = np.array(())
        mayorProb = 0
        eleccion = 0

        for dato in datostest:
            prediccionDato = self.probabilidadClases(dato, atributosDiscretos)
            for clase, probabilidad in prediccionDato.items():
                if(probabilidad > mayorProb):
                    eleccion = clase
                    mayorProb = probabilidad
            predicciones = np.append(predicciones, [eleccion])

        return predicciones

##############################################################################

class ClasificadorVecinosProximos(Clasificador):
    mediasEntrenamiento = []
    desviacionesEntrenamiento = []
    resultadosEntrenamiento = {}
    k = 0
    medias = []
    desviaciones = []
    normalizacion = False
    numClases = 0

    def __init__(self, k, normalizacion):
        self.nombreClasificador = "VecinosProximos"
        self.k = k
        self.normalizacion = normalizacion

    def reset(self):
        self.medias = []
        self.desviaciones = []
        self.resultadosEntrenamiento = {}
        self.numClases = 0

    def calcularMediasDesv(self, datostrain, atributosDiscretos):
        #TODO implement this
        i = 0
        valoresAtributos = None
        self.medias = []
        self.desviaciones = []
        # Generamos una lista de longitud numero de atributos que contiene listas con los valores de cada atributo
        valoresAtributos = zip(*datostrain)
        # Calculamos media y desviacion tipica y lo guardamos en la tabla de resultados
        for atributo in valoresAtributos:
            if(atributosDiscretos[i] == False):
                self.medias.append(self.media(atributo))
                self.desviaciones.append(self.desviacionTipica(atributo))
            i += 1
        return

    def normalizarDatos(self, datos, atributosDiscretos):
        #TODO implement this
        i = 0
        valoresAtributos = None
        self.calcularMediasDesv(datos, atributosDiscretos)
        # Generamos una lista de longitud numero de atributos que contiene listas con los valores de cada atributo
        valoresAtributos = zip(*datos)
        for atributo in valoresAtributos:
            if(atributosDiscretos[i] == False):
                for valor in atributo:
                    valor -= self.medias[i]
                    valor /= self.desviaciones[i]
            i += 1
        
        return datos

    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario):
        
        #igual que el train de Naive Bayes para separar en discretos y continuos
        datosClases = {}
        for i in range(len(datosTrain)):
            if(datosTrain[i][-1] not in datosClases):
                datosClases[datosTrain[i][-1]] = []
                self.numClases += 1
            datosClases[datosTrain[i][-1]].append(datosTrain[i])

        #comprobamos si tenemos que normalizar los datos o no
        if self.normalizacion == True:
            self.resultadosEntrenamiento = self.normalizarDatos(datosTrain, atributosDiscretos)
        else:
            self.resultadosEntrenamiento = datosTrain

        #ya estaria finalizado el entrenamiento, porque hemos guardado los datos train en una lista

    def clasePredominante(self, KDistancias):
        #TODO implement this
        rankingClases = {}
        bestClassCount = 0
        clasePredominante = -1
        for distancia in KDistancias:
            if not distancia[1] in rankingClases:
                rankingClases[distancia[1]] = 1
            else:
                rankingClases[distancia[1]] += 1
        for clase in rankingClases.keys():
            if rankingClases[clase] > bestClassCount:
                clasePredominante = clase
        
        return clasePredominante

    def clasifica(self, datosTest, atributosDiscretos, diccionario):

        kDistancias = []
        distancias = []
        datosTestNormalizados = []
        resultadosClasificador = np.array(())

        #comprobamos si tenemos que normalizar los datos o no (igual que el entrenamiento)
        if self.normalizacion == True:
            datosTestNormalizados = self.normalizarDatos(datosTest, atributosDiscretos)
        else:
            datosTestNormalizados = datosTest

        #calculamos las distancias y guardamos una lista con las k distancias mas pequeñas por cada datoTest
        for datoTest in datosTestNormalizados:
            for datoTrain in self.resultadosEntrenamiento:
                distancias.append([distance.euclidean(datoTest[:-1], datoTrain[:-1]), int(datoTrain[-1])]) #guardamos la distancia euclidea y la clase a la que pertecene el dato test
            distancias.sort()
            kDistancias.append(distancias[:self.k])
            distancias = []

        for distanciasDato in kDistancias:
            #aqui debemos recorrer las k distancias de cada punto, indicar la clase a la que pertecene y luego hacer la media y la desviacion de error, eso esta hecho en Nb peero no se como aprovecharlo para aqui.
        
            #TODO llamar a clasePredominante, la cual calculara la clase para ese dato de test y la devuelve
            #TODO meter la clase resultado en el array de resultados
            resultadosClasificador = np.append(resultadosClasificador, [self.clasePredominante(distanciasDato)])

        return resultadosClasificador


##############################################################################

class ClasificadorRegresionLogistica(Clasificador):
    wFrontera = []
    numEpocas = 0
    cteAprendizaje = 0
    normalizacion = False
    numClases = 0
    medias = []
    desviaciones = []
    numE = 2.71828
    datosTrain = {}

    def __init__(self, numEpocas, cteAprendizaje, normalizacion):
        self.nombreClasificador = "RegresionLogistica"
        self.numEpocas = numEpocas
        self.cteAprendizaje = cteAprendizaje
        self.normalizacion = normalizacion

    def reset(self):
        self.medias = []
        self.desviaciones = []
        self.cteAprendizaje = 0
        self.numEpocas = 0
        self.wFrontera = []
        self.numClases = 0
        self.datosTrain = {}

    def calcularMediasDesv(self, datostrain, atributosDiscretos):
        #TODO implement this
        i = 0
        valoresAtributos = None
        self.medias = []
        self.desviaciones = []
        # Generamos una lista de longitud numero de atributos que contiene listas con los valores de cada atributo
        valoresAtributos = zip(*datostrain)
        # Calculamos media y desviacion tipica y lo guardamos en la tabla de resultados
        for atributo in valoresAtributos:
            if(atributosDiscretos[i] == False):
                self.medias.append(self.media(atributo))
                self.desviaciones.append(self.desviacionTipica(atributo))
            i += 1
        return

    def normalizarDatos(self, datos, atributosDiscretos):
        #TODO implement this
        i = 0
        valoresAtributos = None
        self.calcularMediasDesv(datos, atributosDiscretos)
        # Generamos una lista de longitud numero de atributos que contiene listas con los valores de cada atributo
        valoresAtributos = zip(*datos)
        for atributo in valoresAtributos:
            if(atributosDiscretos[i] == False):
                for valor in atributo:
                    valor -= self.medias[i]
                    valor /= self.desviaciones[i]
            i += 1
        
        return datos

    def calculaGI(self, wFrontera, dato):
        xBarra = dato
        xBarra = np.insert(xBarra, 0, 1)
        a = np.dot(wFrontera, dato)
        if (-a > 700):
            return 0.0
        else:
            return 1/ (1+pow(self.numE,-a))

    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario):
        self.wFrontera = [0] * len(atributosDiscretos)
        random.seed()

         #Inicializamos el vector aleatorio
        for i in range(len(self.wFrontera)):
            self.wFrontera[i] = float(decimal.Decimal(random.randrange(-50, 50))/100)

        #comprobamos si tenemos que normalizar los datos o no
        if self.normalizacion == True:
            self.datosTrain = self.normalizarDatos(datosTrain, atributosDiscretos)
        else:
            self.datosTrain = datosTrain

        #Entrenamos
        for i in range(self.numEpocas):
            for j in range(len(self.datosTrain)):
                gI = self.calculaGI(self.wFrontera, self.datosTrain[j])
                gIMenosClase = gI - self.datosTrain[j][-1]
                for k in range(len(self.wFrontera)):
                    self.wFrontera[k] = self.wFrontera[k] - self.cteAprendizaje * (gIMenosClase * self.datosTrain[j][k]) 

        #ya estaria finalizado el entrenamiento, porque hemos guardado los datos train en una lista

    def clasifica(self, datosTest, atributosDiscretos, diccionario):

        datosTestNormalizados = []
        resultadosClasificador = np.array(())

        #comprobamos si tenemos que normalizar los datos o no (igual que el entrenamiento)
        if self.normalizacion == True:
            datosTestNormalizados = self.normalizarDatos(datosTest, atributosDiscretos)
        else:
            datosTestNormalizados = datosTest

        #calculamos las distancias y guardamos una lista con las k distancias mas pequeñas por cada datoTest
        for datoTest in datosTestNormalizados:
            probC1 = self.calculaGI(self.wFrontera, datoTest)
            if probC1 >= 0.5:
                resultadosClasificador = np.append(resultadosClasificador, 1)
            else:
                resultadosClasificador = np.append(resultadosClasificador, 0)
            probC1 = 0

        return resultadosClasificador


 








    

