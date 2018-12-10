#faltan los imports que ahora mismo no se cuales nos van a hacer falta
import numpy as np


###Clase que es un atributo de un dato (letra)
class Atributo:
    valorAtributo = -1
    nombreAtributo = ""
    tipoAtributo = ""

    def __init__(self, valorAtributo, nombreAtributo, tipoAtributo): 
       self.valorAtributo = valorAtributo
       self.nombreAtributo = nombreAtributo
       self.tipoAtributo = tipoAtributo

    def getValorAtributo(self):
        return self.valorAtributo
    
    def getNombreAtributo(self):
        return self.nombreAtributo

    def getTipoAtributo(self):
        return self.tipoAtributo

    def setValorAtributo(self, valorAtributo):
        self.valorAtributo = valorAtributo
    
    def setNombreAtributo(self, nombreAtributo):
        self.nombreAtributo = nombreAtributo

    def setTipoAtributo(self, tipoAtributo):
        self.tipoAtributo = tipoAtributo

###Clase que es un dato (letra) que se compone de la clase (letra en este caso) y una lista de atributos
class Dato:
    clase = None            #la clase sera un atributo propio con su valor, su tipo y como nombre será clase o numero
    listaAtributos = []     #lista con todos los atributos sin la clase (que la separamos)

    def __init__(self, clase): 
        self.clase = clase

    def añadirAtributoAlDato(self, atributo):
        self.listaAtributos.append(atributo)

class letrita:
    clase = ""
    celda = None

    def __init__(self, clase, celda):

         self.clase = clase
         self.celda = celda

##Funciones de recortar la imagen (copiado de lo de eva)
def recortarImagen(celda):
    
    aux = []
    for linea in celda:
        if False in linea:
            aux.append(linea)

    return np.array(aux)


def recortarLetra(letra):
    recorte = 70
    letraRecortada = (letra>recorte)
    letraRecortada = letraRecortada[10:-10,10:-10]
    letraRecortada = recortarImagen(letraRecortada)
    letraRecortada = np.transpose(letraRecortada(np.transpose(letraRecortada)))

    return letraRecortada

##Funciones que recorta la imagen con N lineas en vertical y horizontal
def cortarLineasVertical(letra, division):

    salto = letra.shape[0]//division
    umbral = [0.6]
    totalLineas = []

    for i in range (0, salto*division,salto):
        lineas=0
        j=0
        aux = np.mean(letra[i:i+salto,:],axis=0)
        aux = (aux>umbral)
        valor = aux[0]

        while j < (len(aux)-1):
            valor = aux[j]
            if not valor:
                lineas += 1
            while aux[j] == valor:
                j += 1
                if j == (len(aux)-1):
                    break
        totalLineas.append(lineas)

    return totalLineas

def cortarLineasHorizontal(letra, division):

    salto = letra.shape[1]//division
    umbral = [0.6]
    totalLineas = []

    for i in range (0, salto*division,salto):
        lineas=0
        j=0
        aux = np.mean(letra[i:i+salto,:],axis=1)
        aux = (aux>umbral)
        valor = aux[0]

        while j < (len(aux)-1):
            valor = aux[j]
            if not valor:
                lineas += 1
            while aux[j] == valor:
                j += 1
                if j == (len(aux)-1):
                    break
        totalLineas.append(lineas)

    return totalLineas














