# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:04:21 2019

@author: ALEXANDERCHF
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


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
# Empezamos a dibujar
plt.figure(figsize=(30,25))

# Obtenemos la correlacion entre los atributos
cor = df.corr()
print('########## CORRELATION BETWEEN FEATURES ##########')
print(cor)

# Ploteamos correlation
sns.heatmap(cor, annot=True, cmap=plt.cm.Blues)
plt.savefig('Correlations.png')

# Correlation de todos los atributos contra el atributo 'SalePrice'
cor_target = abs(cor["SalePrice"])

# Selccionamos los atributos con al menos 0,5 de correlacion
relevant_features = cor_target[cor_target>0.5]

print('########## FEATURES WITH LONG CORRELATION WITH SALEPRICE ##########')
print(relevant_features)

for column in df:
    if column not in relevant_features:
        df = df.drop(column, 1)

print('########## DATASET ( FEATURES WITH LONG CORRELATION WITH SALEPRICE )  ##########')
print(df)


print('########## DATASET ( FEATURES WITH LONG CORRELATION WITH SALEPRICE AND WITHOUT CORRELATION BETWEEN THEM)  ##########')
# Eliminamos los atributos con un alto indice de correlacion
new_df = determinateDataWithOutCorrelation(df, 0.7, "SalePrice")
print(new_df)