import numpy as np

##PUNTOS DE CORTE HORIZONTALES Y VERTICALES

def contarLineasVertical(celda, division):

    salto = celda.shape[0]//division
    umbral = [0.6]
    totalLineas = []

    for i in range (0, salto*division,salto):
        lineas=0
        j=0
        aux = np.mean(celda[i:i+salto,:],axis=0)
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

def contarLineasHorizontal(celda, division):

    salto = celda.shape[1]//division
    umbral = [0.6]
    totalLineas = []
    
    for i in range (0, salto*division,salto):
        lineas=0
        j=0
        aux = np.mean(celda[:,i:i+salto],axis=1)
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

def generacionDatos(letritas, letritasRecortadas, listaAtributos, numDivisionesH=0, numDivisionesV=0, splits=0):

    numAtributos = 1
    
    for atributo in listaAtributos:
        if atributo == True:
            numAtributos += 1
    
    if numDivisionesH != 0:
        numAtributos -= 1
    if numDivisionesV != 0:
        numAtributos -= 1
    if splits != 0:
        numAtributos -= 1
              
    datos = np.zeros((len(letritasRecortadas), (splits**2)+numDivisionesH+numDivisionesV+numAtributos))
    i=0
    for letrita, letritaRecortada in zip(letritas, letritasRecortadas):
        
        if splits!= 0:
            x = letrita.celda.shape[1]//splits
            y = letrita.celda.shape[0]//splits
        
        j = 0
        
        if listaAtributos[0] == True:                               #Atributo Alto
            alto = letritaRecortada.celda.shape[0]
            datos[i][j] = alto
            j+=1
        if listaAtributos[1] == True:                               #Atributo Ancho
            ancho = letritaRecortada.celda.shape[1]
            datos[i][j] = ancho
            j+=1
        if listaAtributos[2] == True:                               #Atributo Relacion
            relacion = letritaRecortada.celda.shape[1]/letritaRecortada.celda.shape[0]
            datos[i][j] = relacion
            j+=1
        if listaAtributos[3] == True:                               #Atributo Pixeles
            pixeles = np.sum(letritaRecortada.celda)/(letritaRecortada.celda.shape[0]*letritaRecortada.celda.shape[1])
            datos[i][j] = pixeles
            j+=1
        if listaAtributos[4] == True:                               #Atributo Cortes H
            cortesH = contarLineasHorizontal(letrita.celda,numDivisionesH)
            for corteH in cortesH:
                datos[i][j] = corteH
                j+=1
        if listaAtributos[5] == True:                               #Atributo Cortes V
            cortesV = contarLineasVertical(letrita.celda,numDivisionesV)
            for corteV in cortesV:
                datos[i][j] = corteV
                j+=1
        if listaAtributos[6] == True:                               #Atributo rejilla
            for n in range(x, letrita.celda.shape[1]+1, x):
                for m in range(y, letrita.celda.shape[0]+1, y):
                    media = np.mean(letrita.celda[m-y:m, n-x:n])
                    datos[i][j] = media
                    j+=1

        datos[i][j] = letrita.clase   
        i+=1

    return datos

def test_datos(self, letritas, letritasRecortadas):
    datos = datos = generacionDatos(letritas, letritasRecortadas, [True, True, True, True, True, True, True], 5, 5, 3)

    print("Matriz de datos obtenida: ")
    print(datos)