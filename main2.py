#Created by: Edwin Murillo & Antonio Alvarez
#Last revision: 11/10/2017

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

    table = pd.read_csv('data2.csv',na_values=['NS','NA','NaN'])
    #Erase position, only the substitution remains
    for i in range(0,len(table.iloc[:,4])):
        table.iloc[i,4] = table.iloc[i,4].translate({ord(c): None for c in '1234567890c.abdefghijklmnopqrstuvwxyz_?+-'})

    #Filling missing values
    table.iloc[:,0].fillna(value=table.iloc[:,0].mean(),inplace=True)
    table.iloc[:,11].fillna(value=table.iloc[:,11].mean(),inplace=True)
    table.iloc[:,15].fillna(value=table.iloc[:,15].mean(),inplace=True)

    #Normalizing
    table.iloc[:,0] = (table.iloc[:,0].mean())/table.iloc[:,0].std()
    table.iloc[:,11] = (table.iloc[:,11].mean())/table.iloc[:,11].std()
    table.iloc[:,15] = (table.iloc[:,15].mean())/table.iloc[:,15].std()

    table = table.dropna(subset=['FATHMM prediction'])

    table = pd.get_dummies(table,dummy_na = True, columns = ['Primary site','Primary histology','Genome-wide screen', 'Mutation CDS','Mutation Description','LOH','Mutation strand','SNP','Resistance Mutation','Mutation somatic status','Sample source','Tumour origin'])

    training_table = table.sample(frac=.8,random_state=22071993) 
    test_table = table[~table.isin(training_table)].dropna(how='all')
    #The feature we want to predict. 
    train_results = training_table['FATHMM prediction']
    test_results = test_table['FATHMM prediction']
    del training_table['FATHMM prediction']
    del test_table['FATHMM prediction']

    #Convert to numpy array for sklearn
    training_table2 = training_table.as_matrix()
    test_table2 = test_table.as_matrix()
    train_results2 = train_results.as_matrix()
    test_results2 = test_results.as_matrix() 

    #Model construction and training
    #Neural network
    param_nn = {'alpha': [0.0001, 0.0005, 0.0009], 'hidden_layer_sizes': [(10,), (30,), (50,), (80,), (100,)], 'max_iter':[1000]}
    nn = MLPClassifier()
    cv_nn = GridSearchCV(nn,param_nn)
    cv_nn.fit(training_table2,train_results2)

    #SVM
    param_svm = {'C': [i for i in range(1,100,20)], 'gamma': ['auto',10, 1, 0.01,], 'kernel': ['rbf','sigmoid']}
    svm = SVC()
    cv_svm = GridSearchCV(svm,param_svm)
    cv_svm.fit(training_table2,train_results2)

    #Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb.fit(training_table2,train_results2)

    #Decision tree
    param_dt = {'criterion': ['entropy'],'max_depth': [None,10,30,50,100],'max_features':['auto','log2',None]}
    dt = DecisionTreeClassifier()
    cv_dt = GridSearchCV(dt,param_dt)
    cv_dt.fit(training_table2,train_results2)

    #Saving the models 
    joblib.dump(cv_nn, 'nn2.pkl')
    joblib.dump(cv_svm, 'svm2.pkl')
    joblib.dump(gnb, 'nb2.pkl')
    joblib.dump(cv_dt, 'dt2.pkl')  


    #Metrics
    predicted_nn = cv_nn.predict(test_table2)
    predicted_svm = cv_svm.predict(test_table2)
    predicted_nb = gnb.predict(test_table2)
    predicted_dt = cv_dt.predict(test_table2)

    conf_matrix_nn = mt.confusion_matrix(test_results2,predicted_nn)
    conf_matrix_svm = mt.confusion_matrix(test_results2,predicted_svm)
    conf_matrix_nb = mt.confusion_matrix(test_results2,predicted_nb)
    conf_matrix_dt = mt.confusion_matrix(test_results2,predicted_dt)

    accuracy_nn = mt.accuracy_score(test_results2,predicted_nn)
    accuracy_svm = mt.accuracy_score(test_results2,predicted_svm)
    accuracy_nb = mt.accuracy_score(test_results2,predicted_nb)
    accuracy_dt = mt.accuracy_score(test_results2,predicted_dt)

    precision_nn = mt.precision_score(test_results2,predicted_nn,pos_label='PATHOGENIC')
    precision_svm = mt.precision_score(test_results2,predicted_svm,pos_label='PATHOGENIC')
    precision_nb = mt.precision_score(test_results2,predicted_nb,pos_label='PATHOGENIC')
    precision_dt = mt.precision_score(test_results2,predicted_dt,pos_label='PATHOGENIC')

    recall_nn = mt.recall_score(test_results2,predicted_nn,pos_label='PATHOGENIC')
    recall_svm = mt.recall_score(test_results2,predicted_svm,pos_label='PATHOGENIC')
    recall_nb = mt.recall_score(test_results2,predicted_nb,pos_label='PATHOGENIC')
    recall_dt = mt.recall_score(test_results2,predicted_dt,pos_label='PATHOGENIC')

    #Roc Curve
    prob_nn= np.matrix(cv_nn.predict_proba(test_table2))[:,0]
    prob_svm= cv_svm.decision_function(test_table2)
    prob_nb= np.matrix(gnb.predict_proba(test_table2))[:,0]
    prob_dt= np.matrix(cv_dt.predict_proba(test_table2))[:,0]
    tpr_nn, fpr_nn,_ = mt.roc_curve(test_results2,prob_nn,pos_label='PATHOGENIC')
    tpr_svm, fpr_svm,_ = mt.roc_curve(test_results2,prob_svm,pos_label='NEUTRAL')
    tpr_nb, fpr_nb,_ = mt.roc_curve(test_results2,prob_dt,pos_label='PATHOGENIC')
    tpr_dt, fpr_dt,_ = mt.roc_curve(test_results2,prob_dt,pos_label='PATHOGENIC')

    #showing metrics
    print("The accuracy for the classifiers are\n")
    print("neural network: ", accuracy_nn)
    print("SVM: ", accuracy_svm)
    print("Naive Bayes: ", accuracy_nb)
    print("Decision tree: ", accuracy_dt)
    print("\n")
    print("The precision for the classifiers are\n")
    print("neural network: ", precision_nn)
    print("SVM: ", precision_svm)
    print("Naive Bayes: ", precision_nb)
    print("Decision tree: ", precision_dt)
    print("\n")
    print("The recall for the classifiers are\n")
    print("neural network: ", recall_nn)
    print("SVM: ", recall_svm)
    print("Naive Bayes: ", recall_nb)
    print("Decision tree: ", recall_dt) 

    print("\n The confusion matrix for the neural network is:")
    print(conf_matrix_nn)
    print("\n The confusion matrix for the neural network is:")
    print(conf_matrix_svm)
    print("\n The confusion matrix for naive bayes is:")
    print(conf_matrix_nb)
    print("\n The confusion matrix for the decision tree is:")
    print(conf_matrix_dt)

    plt.figure()
    plt.plot(fpr_nn,tpr_nn, color='darkorange',lw=2,label='ROC curve')
    plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC of neural network')
    plt.legend(loc="lower right")
    plt.savefig('roc_nn2.png')
 
    plt.figure()
    plt.plot(fpr_svm,tpr_svm, color='darkorange',lw=2,label='ROC curve')
    plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC of SVM')
    plt.legend(loc="lower right")
    plt.savefig('roc_svm2.png')

    plt.figure()
    plt.plot(fpr_nb,tpr_nb, color='darkorange',lw=2,label='ROC curve')
    plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC of Naive Bayes')
    plt.legend(loc="lower right")
    plt.savefig('roc_nb2.png')

    plt.figure()
    plt.plot(fpr_dt,tpr_dt, color='darkorange',lw=2,label='ROC curve')
    plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC of Decision Tree')
    plt.legend(loc="lower right")
    plt.savefig('roc_dt2.png')
