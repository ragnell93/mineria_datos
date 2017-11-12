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

    table3 = pd.read_csv('data1Somatic_Germline.csv',na_values=['NA','NaN'])
    del table3['Type']
    table2 = pd.read_csv(sys.argv[1],na_values=['NA','NaN'])
    table = pd.concat([table3,table2])
    #Filling missing values for numeric types with mean
    table.iloc[:,1].fillna(value=table.iloc[:,1].mean(),inplace=True)
    table.iloc[:,2].fillna(value=table.iloc[:,1].mean(),inplace=True)

    #convert nominal variables to binary array
    table = pd.get_dummies(table,dummy_na = True, columns = ['ExonIntron', 'Description', 'WT_nucleotide','Mutant_nucleotide', 'CpG_site','Splice_site','WT_codon','Mutant_codon','WT_AA','Mutant_AA','Effect','SIFTClass','Polyphen2','TransactivationClass','DNEclass','Short_topo','Morphology'])
    table.iloc[:,0] = (table.iloc[:,0].mean())/table.iloc[:,0].std()
    table.iloc[:,1] = (table.iloc[:,1].mean())/table.iloc[:,1].std()

    predict1 = table.iloc[6671:,:]

    clf = joblib.load('svm1.pkl') 

    predict2 = predict1.as_matrix()

    y_pred = clf.predict(predict2)
    print(y_pred)