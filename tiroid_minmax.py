import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics


dataset = pd.read_csv("new_data_tiroid.csv")
dataset.columns = ['a','b','c','d','e','label']
X = dataset.iloc[:, :-1]
y = dataset['label']

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

loo = LeaveOneOut()
i=1 

for train_index, test_index in loo.split(X):
    print("Loo ", i)
    print("TRAIN :", train_index, "TEST :", test_index)
    X_train=X[train_index]
    X_test=X[test_index]
    y_train=y[train_index]
    y_test=y[test_index]
    i+=1

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X,y)
print(knn)

scores = cross_val_score(knn, X, y, cv=loo, scoring='accuracy') 
print(scores)        
print("Accuracy", scores.mean())
print("Error Ratio Loo : ", 1-scores.mean())
