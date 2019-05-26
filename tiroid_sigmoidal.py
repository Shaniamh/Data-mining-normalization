import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics


data = pd.read_csv('new_data_tiroid.csv', header=None)
# data
data.columns = ['a','b','c','d','e','label']
# split between data and class target
data_valuesig = data[['a','b','c','d','e']]
data_targetsig = data[['label']].astype(int)

#rumus
std=data.std()
x = (data-data.mean())/std
# sigmoidal = ((1-np.exp(-x))/(1+np.exp(-x)))
datazscore = pd.DataFrame(data_valuesig.apply(lambda x: (1-np.exp(-x))/(1+np.exp(-x)), axis=0))
datazscore

Xz = data_valuesig[['a', 'b', 'c', 'd', 'e']]
yz = data_targetsig

loo = LeaveOneOut()
i=1 

for train_index, test_index in loo.split(Xz):
    print("Loo ", i)
    print("TRAIN :", train_index, "TEST :", test_index)
    Xz_train=Xz.iloc[train_index]
    Xz_test=Xz.iloc[test_index]
    yz_train=yz.iloc[train_index]
    yz_test=yz.iloc[test_index]
    i+=1

knnz = KNeighborsClassifier(n_neighbors=3)
knnz.fit(Xz_train,yz_train)
print(knnz)

scoresz = cross_val_score(knnz, Xz, yz, cv=loo, scoring='accuracy') 
print(scoresz)        
print("Accuracy", scoresz.mean())
print("Error Ratio Loo : ", 1-scoresz.mean())