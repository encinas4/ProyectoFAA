from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

def clasificaNaiveBayes(datos, divisionVC):
    clasificador = GaussianNB()
    
    validacionCruzada = KFold(n_splits = divisionVC, shuffle = True)
    aciertos = cross_val_score(clasificador, datos[:,:-1], datos[:,-1], cv = validacionCruzada)
    
    mediaAciertos = np.mean(aciertos)
    
    return mediaAciertos

def clasificaKNN(datos, k, divisionVC):
    clasificador =  KNeighborsClassifier(n_neighbors=k,weights='uniform')
    
    validacionCruzada = KFold(n_splits = divisionVC, shuffle = True)
    aciertos = cross_val_score(clasificador, datos[:,:-1], datos[:,-1], cv = validacionCruzada)
    
    mediaAciertos = np.mean(aciertos)
    
    return mediaAciertos

def clasificaRegresionLogistica(datos, nEpocas, divisionVC):
    clasificador = LogisticRegression(max_iter = nEpocas)
    
    validacionCruzada = KFold(n_splits = divisionVC, shuffle = True)
    aciertos = cross_val_score(clasificador, datos[:,:-1], datos[:,-1], cv = validacionCruzada)
    
    mediaAciertos = np.mean(aciertos)
    
    return mediaAciertos

def clasificaRandomForest(datos, estimadores,  minimoEjemplos, divisionVC):
    clasificador = RandomForestClassifier(n_estimators = estimadores, max_depth=None, min_samples_split= minimoEjemplos, random_state=0)
    
    validacionCruzada = KFold(n_splits = divisionVC, shuffle = True)
    aciertos = cross_val_score(clasificador, datos[:,:-1], datos[:,-1], cv = validacionCruzada)
    
    mediaAciertos = np.mean(aciertos)
    
    return mediaAciertos


