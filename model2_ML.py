import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier



def popular_prediction(data):

    # def split_data(data1):
        
    #     X = data1.drop(columns =['popularity',data1.columns[0]])
    #     y = data1.loc[:,'popularity']

    #     # Train-Test slpit of 70%-30%
    #     X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3, stratify=y, random_state=102)




        # return (X_train, X_test, y_train, y_test)

    # def base_learners_evaluations(data1):
    #     X_train, X_test, y_train, y_test = split_data(data1)
        
    #     idx = []
    #     scores = {'F1 score': [], 'Recall':[]}
    #     for bc in base_classifiers:
    #         lr = bc[1]
    #         lr.fit(X_train, y_train)

    #         predictions = lr.predict(X_test)

    #         idx.append(bc[0])
    #         scores['F1 score'].append(metrics.f1_score(y_test, predictions))
    #         scores['Recall'].append(metrics.recall_score(y_test, predictions))

    # def ensemble_evaluation(data1, model, label='Original'):
    #     X_train, X_test, y_train, y_test = split_data(data1)
    #     model.fit(X_train, y_train)
    #     predictions = model.predict(X_test)       
    #     #return pd.DataFrame({'F1 Score': [metrics.f1_score(y_test, predictions)],
    #                         #'Recall': [metrics.recall_score(y_test, predictions)]}, index=[label])
    #     # print(X_test.shape,X_train.shape)
    #     return model

#Approach1: 
    # data1 = pd.read_csv('/Users/user/Desktop/Post_image/posts_image/final_img.csv')

    # ensemble = RandomForestClassifier(n_estimators=80, criterion="entropy", n_jobs=-1)
    
    #X = data.drop(columns =['popularity',data.columns[0]])
    #y = data.loc[:,'popularity']

    # ensemble.fit(X,y)

    # return ensemble
#Approach2 (load_model):
    loaded_model = pickle.load(open('/Users/user/Desktop/Post_image/posts_image/rf_model.sav', 'rb'))
    result = loaded_model.predict(data)

    return result

    #ensemble = ensemble_evaluation(data1, ensemble, label='Original')

    #return model.predict()   

    #predictions_1 = ensemble.predict_proba(data)[:,1]
    
