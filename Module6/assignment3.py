#
# This code is intentionally missing!
# Read the directions on the course lab page!
#
import pandas as pd
import numpy as np

X = pd.read_csv(r'E:\Python_SYY\edx\DAT210x\Module6\Datasets\parkinsons.data')
X = X.drop(axis = 1, labels=['name'])
y = X['status']
X = X.drop(axis = 1, labels=['status'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state = 7)

from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
model.score(X_test, y_test)

C_range = np.arange(0.05,2,0.05)
gamma_range = np.arange(0.001,1,0.001)
best_score = 0
for C in C_range:
    for gamma in gamma_range:
        model = SVC(C = C, gamma = gamma)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        if score > best_score:
            best_score = score
            
# transforming the data(Normalizer)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state = 7)
from sklearn.preprocessing import Normalizer
norm = Normalizer()
norm.fit(X_train)
X_train = norm.transform(X_train)
X_test = norm.transform(X_test)
best_score_norm = 0
for C in C_range:
    for gamma in gamma_range:
        model = SVC(C = C, gamma = gamma)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        if score > best_score_norm:
            best_score_norm = score
            
# transforming the data (MaxabsScaler())
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state = 7)
from sklearn.preprocessing import MaxAbsScaler
norm = MaxAbsScaler()
norm.fit(X_train)
X_train = norm.transform(X_train)
X_test = norm.transform(X_test)
best_score_max = 0
for C in C_range:
    for gamma in gamma_range:
        model = SVC(C = C, gamma = gamma)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        if score > best_score_max:
            best_score_max = score
            
# transforming the data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state = 7)
from sklearn.preprocessing import  MinMaxScaler
norm =  MinMaxScaler()
norm.fit(X_train)
X_train = norm.transform(X_train)
X_test = norm.transform(X_test)
best_score_min = 0
for C in C_range:
    for gamma in gamma_range:
        model = SVC(C = C, gamma = gamma)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        if score > best_score_min:
            best_score_min = score
            
# transforming the data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state = 7)
from sklearn.preprocessing import KernelCenterer
norm = KernelCenterer()
norm.fit(X_train)
X_train = norm.transform(X_train)
X_test = norm.transform(X_test)
best_score_ker = 0
for C in C_range:
    for gamma in gamma_range:
        model = SVC(C = C, gamma = gamma)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        if score > best_score_ker:
            best_score_ker = score
            
# transforming the data(StandardScaler)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state = 7)
from sklearn.preprocessing import StandardScaler
norm = StandardScaler()
norm.fit(X_train)
X_train = norm.transform(X_train)
X_test = norm.transform(X_test)
best_score_st = 0
for C in C_range:
    for gamma in gamma_range:
        model = SVC(C = C, gamma = gamma)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        if score > best_score_st:
            best_score_st = score
