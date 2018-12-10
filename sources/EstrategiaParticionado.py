from abc import ABCMeta, abstractmethod
from Datos import Datos
import random
import numpy as np


class Particion:

    indicesTrain = []
    indicesTest = []

    def __init__(self, train, test):
        self.indicesTrain = train
        self.indicesTest = test

#####################################################################################################


class EstrategiaParticionado:

    # Clase abstracta
    __metaclass__ = ABCMeta

    # Atributos: deben rellenarse adecuadamente para cada estrategia concreta
    nombreEstrategia = "null"
    numeroParticiones = 0
    particiones = []

    @abstractmethod
    # TODO: esta funcion deben ser implementadas en cada estrategia concreta
    def creaParticiones(self, datos, seed=None):
        pass


#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):

    # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado.
    # Devuelve una lista de particiones (clase Particion)
    # TODO: implementar

    divPercent = 0

    def __init__(self, divPercent, numPart):
        self.divPercent = divPercent
        self.nombreEstrategia = "ValidacionSimple"
        self.numeroParticiones = numPart

    def creaParticiones(self, datos, seed=None):
        random.seed(seed)
        indices = []
        indicesPermutados = []
        self.particiones = []
        train = []
        test = []

        # new loop for N partition
        # in each iteration, permutation the data and create new partition
        i=0
        for i in range(datos.numDatos):
            indices.append(i)

        for i in range(self.numeroParticiones):
            indicesPermutados = np.random.permutation(indices)
            frontier = int(len(indicesPermutados) * self.divPercent)
            train = indicesPermutados[:frontier]
            particion = Particion(indicesPermutados[:frontier].tolist(), indicesPermutados[frontier:].tolist())
            self.particiones.append(particion)

        return self.particiones


#####################################################################################################
class ValidacionCruzada(EstrategiaParticionado):

    # Crea particiones segun el metodo de validacion cruzada.
    # El conjunto de entrenamiento se crea con las nfolds-1 particiones
    # y el de test con la particion restante
    # Esta funcion devuelve una lista de particiones (clase Particion)
    # TODO: implementar

    def __init__(self, numPart):
        self.numeroParticiones = numPart
        self.nombreEstrategia = "ValidacionCruzada"

    def creaParticiones(self, datos, seed=None):
        indices = []
        indicesPermutados = []
        partitionSize = int(datos.numDatos / self.numeroParticiones)
        i = 0
        j = 0
        k = partitionSize
        self.particiones = []
        auxLists = []

        for i in range(datos.numDatos):
            indices.append(i)

        indicesPermutados = np.random.permutation(indices)

        i=0
        # Create folds
        for j in range(self.numeroParticiones):
            auxLists.append(indicesPermutados[i:k])
            i += partitionSize
            k += partitionSize

        # Create partitions from folds
        for i in range(len(auxLists)):
            trainList = []
            for j in range(len(auxLists)):
                if(j != i):
                    trainList.append(auxLists[j])
            testList = auxLists[i]
            partition = Particion(
                [item for sublist in trainList for item in sublist], testList.tolist())

            self.particiones.append(partition)

        return self.particiones

#####################################################################################################


class ValidacionBootstrap(EstrategiaParticionado):

    def __init__(self, numPart):
        self.nombreEstrategia = "ValidacionBootstrap"
        self.numeroParticiones = numPart

    # Crea particiones segun el metodo de boostrap
    # Devuelve una lista de particiones (clase Particion)
    # TODO: implementar
    def creaParticiones(self, datos, seed=None):
        train = []
        test = []
        self.particiones = []
        random.seed(seed)
        indices = []
        i=0
        for i in range(datos.numDatos):
            indices.append(i)

        # creamos N particiones segun se nos indice en el constructor. En cada iteracion:
        for i in range(self.numeroParticiones):
            for j in range(datos.numDatos):
                indiceRandom = random.randint(0, datos.numDatos - 1)
                train.append(indiceRandom)
            for l in indices:
                if not l in train:  
                    test.append(l)
            partition = Particion(train, test)  
            self.particiones.append(partition)
            train = []
            test = []

        return self.particiones

def main():
    dataset=Datos('ConjuntosDatos/balloons.data')
    estrategia1 = ValidacionSimple(.6, 4)
    estrategia2 = ValidacionCruzada(4)
    estrategia3 = ValidacionBootstrap(4)
    particiones1 = estrategia1.creaParticiones(dataset)
    particiones2 = estrategia2.creaParticiones(dataset)
    particiones3 = estrategia3.creaParticiones(dataset)
    print("\n\n\t\t\tValidacion Simple")
    i=0
    for particion in particiones1:
        print("Partición numero:", i+1)
        print("Train:")
        print(particion.indicesTrain)
        print("Test:")
        print(particion.indicesTest)
        i +=1

    print("\n\n\t\t\tValidacion Cruzada")
    i=0
    for particion in particiones2:
        print("Partición numero:", i+1)
        print("Train:")
        print(particion.indicesTrain)
        print("Test:")
        print(particion.indicesTest)
        i +=1

    print("\n\n\t\t\tValidacion Boostrap")
    i=0
    for particion in particiones3:
        print("Partición numero:", i+1)
        print("Train:")
        print(particion.indicesTrain)
        print("Test:")
        print(particion.indicesTest)
        i +=1





if __name__ == '__main__':
    main()