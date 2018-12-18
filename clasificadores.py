from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np

#from keras.models import Sequential
#from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


def clasificaNaiveBayes(datos, divisionVC):
    clasificador = GaussianNB(priors=None)
    
    validacionCruzada = StratifiedKFold(n_splits = divisionVC, random_state=None, shuffle=True)
    aciertos = cross_val_score(clasificador, datos[:,:-1], datos[:,-1], cv = validacionCruzada)
    
    mediaAciertos = np.mean(aciertos)
    
    return mediaAciertos

def clasificaKNN(datos, k, divisionVC):
    clasificador =  KNeighborsClassifier(n_neighbors=k, weights='distance', p=1, algorithm='auto')
    
    validacionCruzada = StratifiedKFold(n_splits = divisionVC, random_state=None, shuffle=True)
    aciertos = cross_val_score(clasificador, datos[:,:-1], datos[:,-1], cv = validacionCruzada)
    
    mediaAciertos = np.mean(aciertos)
    
    return mediaAciertos

def clasificaRegresionLogistica(datos, nEpocas, divisionVC):
    clasificador = LogisticRegression(max_iter=nEpocas, class_weight="balanced", solver="newton-cg",multi_class="multinomial")
    
    validacionCruzada = StratifiedKFold(n_splits = divisionVC, random_state=None, shuffle=True)
    aciertos = cross_val_score(clasificador, datos[:,:-1], datos[:,-1], cv = validacionCruzada)
    
    mediaAciertos = np.mean(aciertos)
    
    return mediaAciertos

def clasificaRandomForest(datos, numArboles,  minimoEjemplosDividir, minimoEjemplosNodo, divisionVC):
    clasificador = RandomForestClassifier(n_estimators = numArboles, criterion="gini", min_samples_split=minimoEjemplosDividir, min_samples_leaf=minimoEjemplosNodo,random_state=0, bootstrap=False)
    
    validacionCruzada = StratifiedKFold(n_splits = divisionVC, random_state=None, shuffle=True)
    aciertos = cross_val_score(clasificador, datos[:,:-1], datos[:,-1], cv = validacionCruzada)
    
    mediaAciertos = np.mean(aciertos)
    
    return mediaAciertos

def clasificaRedNeuronal(datos, numeroIteraciones, divisionVC):
    clasificador = MLPClassifier(activation='tanh', solver='adam', learning_rate='invscaling', max_iter=numeroIteraciones)
    
    validacionCruzada = StratifiedKFold(n_splits = divisionVC, random_state=None, shuffle=True)
    aciertos = cross_val_score(clasificador, datos[:,:-1], datos[:,-1], cv = validacionCruzada)
    
    mediaAciertos = np.mean(aciertos)
    
    return mediaAciertos

def clasificaArbolDecision(datos, divisionVC):
    clasificador = DecisionTreeClassifier(criterion='entropy')
    
    validacionCruzada = StratifiedKFold(n_splits = divisionVC, random_state=None, shuffle=True)
    aciertos = cross_val_score(clasificador, datos[:,:-1], datos[:,-1], cv = validacionCruzada)
    
    mediaAciertos = np.mean(aciertos)
    
    return mediaAciertos


# def clasificaRedNeuronalNuevo(datos, numeroIteraciones, divisionVC):
#     model = Sequential()
#     model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Flatten())
#     model.add(Dense(128, activation=tf.nn.relu))
#     model.add(Dropout(0.2))
#     model.add(Dense(10,activation=tf.nn.softmax))
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     model.fit(x=datos[:,:-1],y=datos[:,-1], epochs=numeroIteraciones)
    
    
#     return model.evaluate(x=datos[:,:-1],y=datos[:,-1])