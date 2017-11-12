#Created by: Edwin Murillo & Antonio Alvarez
#Last revision: 11/10/2017

import sys
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as mt
from sklearn.externals import joblib



if __name__ == '__main__':

    table3 = pd.read_csv('data2.csv',na_values=['NS','NA','NaN'])
    table3 = table3.dropna(subset=['FATHMM prediction'])
    del table3['FATHMM prediction']
    table2 = pd.read_csv(sys.argv[1],na_values=['NS','NA','NaN'])
    table = pd.concat([table3,table2])
    for i in range(0,len(table.iloc[:,4])):
        table.iloc[i,4] = table.iloc[i,4].translate({ord(c): None for c in '1234567890c.abdefghijklmnopqrstuvwxyz_?+-'})

    #Filling missing values
    table.iloc[:,0].fillna(value=table.iloc[:,0].mean(),inplace=True)
    table.iloc[:,10].fillna(value=table.iloc[:,10].mean(),inplace=True)
    table.iloc[:,14].fillna(value=table.iloc[:,14].mean(),inplace=True)

    #Normalizing
    table.iloc[:,0] = (table.iloc[:,0].mean())/table.iloc[:,0].std()
    table.iloc[:,10] = (table.iloc[:,10].mean())/table.iloc[:,10].std()
    table.iloc[:,14] = (table.iloc[:,14].mean())/table.iloc[:,14].std()

    table = pd.get_dummies(table,dummy_na = True, columns = ['Primary site','Primary histology','Genome-wide screen', 'Mutation CDS','Mutation Description','LOH','Mutation strand','SNP','Resistance Mutation','Mutation somatic status','Sample source','Tumour origin'])


    predict1 = table.iloc[2747:,:]

    clf = joblib.load('svm2.pkl') 

    predict2 = predict1.as_matrix()
    y_pred = clf.predict(predict2)
    print(y_pred)
