# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:04:21 2019

@author: ALEXANDERCHF
"""

import pandas as pd
#pd.set_option('display.max_rows', 2000)
#pd.set_option('display.max_columns', 500)
#pd.set_option('display.width', 1000)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import random
from statistics import mode
from collections import Counter


def generateSubDataSet(data):
    features_to_delete = []
    for column in data:
        if random.randint(0, 1) == 0:  # Delete column
            features_to_delete = np.append(features_to_delete, column)
            
    subdataset = data.drop(features_to_delete, 1)
    return subdataset
    
def getInconsistency(subdataset, data):
#    print(subdataset)
#    subset_copy = pd.DataFrame(data = subdataset)
    
    
    features = subdataset.drop('Id', 1).columns
#    for column in subdataset:
#        if column != 'Id':
#            features = np.append(column, features)
#    for column in subdataset:
#        if column != 'Id':
#            features = np.append(features, column)
    print('features: ', features)
    subdataset['is_duplicated'] = subdataset.duplicated(features)
    n = subdataset['is_duplicated'].sum()
    print('n: ', n)
    classes_n = []
    for index, row in subdataset.iterrows():
        if row['is_duplicated'] == True:
            idx = row['Id']
            print('idx: ', idx)
            current_class = data.loc[ idx - 1 , "SalePrice"]
            classes_n = np.append(current_class, classes_n)
    
#    print('classes_n: ', classes_n)
#    unique_elements, counts_elements = np.unique(classes_n, return_counts=True)
#    print(np.asarray((counts_elements.max())))
#    print(Counter(classes_n).most_common()[0][0])
#    print('mode classes_n: ', mode(classes_n))
             
#    with open("Output.txt", "w") as text_file:
#        print(subdataset, file=text_file)
    
#    print(subdataset)
#    for index, row in subdataset.iterrows():
#        if row['is_duplicated'] == True:
#            print(row)

def LVF(data, max_tries, allow_inconsistency):
    n_features      = data.shape[0]
    best_n_features = n_features
    best_subdataset = []
    for i in range(max_tries):        
        current_subdataset = generateSubDataSet(data)        
        current_n_features = current_subdataset.shape[0]        
        if current_n_features < best_n_features:
            if getInconsistency() < allow_inconsistency:
                best_subdataset = current_subdataset
                best_n_features = current_n_features            
        elif current_n_features == best_n_features and getInconsistency() < allow_inconsistency:
            best_subdataset = current_subdataset
    return best_subdataset
        

def determinateDataWithOutCorrelation(data, index, column_class):
     corr_data = data.corr()         
     corr_data = corr_data.where(np.triu(np.ones(corr_data.shape), k = 1).astype(np.bool))        
     to_drop   = [column for column in corr_data.columns if any(corr_data[column].abs() > index)]
     
     print('FEATURES TO DELETE')
     print(to_drop)
     new_data  = pd.DataFrame(data = data)
     
     for column in new_data:
         if column in to_drop  and column_class != column:
             new_data = new_data.drop(column, 1)
     
     return new_data
     
# Cargamos dataset
df = pd.read_csv('train.csv')

subdataset = generateSubDataSet(df.drop('SalePrice', axis = 1))
subdataset['Id'] = df['Id']
#print('subdataset')
#print(subdataset)

getInconsistency(subdataset, df)


# Empezamos a dibujar
#plt.figure(figsize=(30,25))
#
## Obtenemos la correlacion entre los atributos
#cor = df.corr()
#print('########## CORRELATION BETWEEN FEATURES ##########')
#print(cor)
#
## Ploteamos correlation
#sns.heatmap(cor, annot=True, cmap=plt.cm.Blues)
#plt.savefig('Correlations.png')
#
## Correlation de todos los atributos contra el atributo 'SalePrice'
#cor_target = abs(cor["SalePrice"])
#
## Selccionamos los atributos con al menos 0,5 de correlacion
#relevant_features = cor_target[cor_target>0.5]
#
#print('########## FEATURES WITH LONG CORRELATION WITH SALEPRICE ##########')
#print(relevant_features)
#
#for column in df:
#    if column not in relevant_features:
#        df = df.drop(column, 1)
#
#print('########## DATASET ( FEATURES WITH LONG CORRELATION WITH SALEPRICE )  ##########')
#print(df)
#
#
#print('########## DATASET ( FEATURES WITH LONG CORRELATION WITH SALEPRICE AND WITHOUT CORRELATION BETWEEN THEM)  ##########')
## Eliminamos los atributos con un alto indice de correlacion
#new_df = determinateDataWithOutCorrelation(df, 0.7, "SalePrice")
#print(new_df)