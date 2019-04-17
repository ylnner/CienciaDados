import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
from sklearn.metrics import confusion_matrix


# Obtiene distancia entre dos puntos x1 y x2
def my_mse( x1, x2):
    count = 0
    for i in range( 0, x1.shape[0]):
        count += (x1[i] - x2[i])*(x1[i] - x2[i])
    return np.sqrt(count)

def most_frequent(List): 
    counter = 0
    num = List[0] 
      
    for i in List: 
        curr_frequency = List.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
  
    return num 


#####################################################################
# k = Quantidade de vecinos
# x = exemplo de prediçao
# X = dataset com somente os atributos
# Y = as classes de predição
def knn(K, x, X, Y):
    # Calculando distancias con otros puntos
    dist = []
    for i in range(X.shape[0]):
        dist.append( [ i, my_mse( x, X[i,:] ) ] )
    
    # Ordenando dist crecientemente
    dist = np.array(dist)
#     print('dist: ', dist)
    dist = dist[dist[:,1].argsort()]
    
    res = []
    for r in range(K):
        res.append(Y[int(dist[r,0])])
    return res, dist

def variarValorK():

    # Cargando datos
    iris = pd.read_csv('bezdekIris.data')
    iris_data = iris.values

    # Se guardan los atributos en X
    X = iris_data[:,0:4]
    # Se guardan las clases en Y
    Y = iris_data[:,4]


    # Variando K
    print('############# K = 3 #############')
    # Escolhemos um elemento para fazer predição
    x = X[19,:]
    K = 3
    res, dist = knn(K, x, X, Y)
    print('Resultado: ', res)
    print('Distância: ', dist[:K])

    print('############# K = 5 #############')
    # Escolhemos um elemento para fazer predição
    x = X[49,:]
    K = 5
    res, dist = knn(K, x, X, Y)
    print('Resultado: ', res)
    print('Distância: ', dist[:K])

    print('############# K = 7 #############')
    # Escolhemos um elemento para fazer predição
    x = X[140,:]
    K = 7
    res, dist = knn(K, x, X, Y)
    print('Resultado: ', res)
    print('Distância: ', dist[:K])

def randomSubSampling(test_size):    
    iris = pd.read_csv('bezdekIris.data')
    iris_data = iris.values    

    n_times_random_subsampling = 10
    for i in range(n_times_random_subsampling):
        # iris_test_data é dataset con que eu vou fazer a teste
        iris_test = iris.sample(n = test_size, replace=True)
        iris_test_data = iris_test.values

        # iris_train_data é dataset onde eu vou fazer a busqueda de distancia
        iris_train = iris.drop(iris_test.index.values)
        iris_train_data = iris_train.values

        # Se guardan los atributos en X
        X_train = iris_train_data[:,0:4]

        # Se guardan las clases en Y
        Y_train = iris_train_data[:,4]

        K = 5
        X_test = iris_test_data[:,0:4]
        Y_test =iris_test_data[:,4]
        Y_pred = []
        print('############################# ', i+1 , '-vez #############################')
        for j in range(iris_test_data.shape[0]):
            x = X_test[j,:]    
            res, dist = knn(K, x, X_train, Y_train)                
            Y_pred.append(most_frequent(res))
                    
        print('CONFUSION MATRIX')
        print(confusion_matrix(Y_test, Y_pred))        
        
def kFoldCrossValidation(k_value):
     iris = pd.read_csv('bezdekIris.data')
     iris_data = iris.values
            
     n_times = math.ceil(iris_data.shape[0]/k_value)
        
        
     for i in range(k_value):
         idx = n_times * i            
         # iris_test_data é dataset con que eu vou fazer a teste   
         iris_test = iris.iloc[idx:idx+n_times,:]
         iris_test_data = iris_test.values
        
        # iris_train_data é dataset onde eu vou fazer a busqueda de distancia
         iris_train = iris.drop(iris_test.index.values)
         iris_train_data = iris_train.values
                        
        # Se guardan los atributos en X
         X_train = iris_train_data[:,0:4]
    
        # Se guardan las clases en Y
         Y_train = iris_train_data[:,4]
    
         K = 5
         X_test = iris_test_data[:,0:4]
         Y_test =iris_test_data[:,4]
         Y_pred = []
         print('############################# ', i+1 , '-vez #############################')
         for j in range(iris_test_data.shape[0]):
            x = X_test[j,:]    
            res, dist = knn(K, x, X_train, Y_train)                
            Y_pred.append(most_frequent(res))
            
         print('CONFUSION MATRIX')
         print(confusion_matrix(Y_test, Y_pred))
        

def leaveOneOut():
    iris = pd.read_csv('bezdekIris.data')
    iris_data = iris.values
    k_value = iris_data.shape[0]
    kFoldCrossValidation(k_value)

variarValorK()
test_size = 40
randomSubSampling(test_size)
k_fold = 10
kFoldCrossValidation(k_fold)
leaveOneOut()

