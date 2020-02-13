"""
modesl of models:
savd models need to be reported

Classisfiers:
Random forest
naive bayes
ridge
perceptron
log regression
log l1

"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split


#%%

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
#%%
path1="/Users/ismailaslan/Desktop/Python/NLPDisaster/my_models_02/"
X_df = pd.read_csv(path1 + "model_preds.csv")
cols=X_df.columns[1:-1]
X=X_df[cols]
y=X_df["target"]

path1="/Users/ismailaslan/Desktop/Python/NLPDisaster/my_models_02/"
test_df = pd.read_csv(path1 + "test_model_preds.csv")

X_test=test_df[cols]
y_test=test_df["target"]

X_train, X_val, y_train, y_val = train_test_split(X, y,random_state=42)

#%%
start_time=time.process_time()

tr_f1, val_f1, test_f1= log_l1(X, X_train, X_val, y_train, y_val, X_test,y_test)
tr_f1l, val_f1l, test_f1l = logregmodel(X, X_train, X_val, y_train, y_val, X_test,y_test)
tr_f1nb, val_f1nb, test_f1nb = naive_ba(X, X_train, X_val, y_train, y_val, X_test,y_test)
tr_f1rd, val_f1rd, test_f1rd = ridge(X, X_train, X_val, y_train, y_val, X_test,y_test)
tr_f1pr, val_f1pr, test_f1pr = perceptr(X, X_train, X_val, y_train, y_val, X_test,y_test)
tr_f1rf, val_f1rf, test_f1rf = rand_forest(X, X_train, X_val, y_train, y_val, X_test,y_test)

print("--- %0.2f seconds ---" % (time.process_time() - start_time))

#%%
def rand_forest(X, X_train, X_val, y_train, y_val, X_test,y_test):
    
    model = RandomForestClassifier(max_depth=2, random_state=0)
    model.fit(X_train, y_train)

    sh=X.shape
    save_model(model,sh)

    tr_f1, val_f1, test_f1, tr_acc, val_acc,test_acc=model_performance(model, X_train, y_train, X_val, y_val, X_test, y_test)
    return tr_f1, val_f1, test_f1

#%%
def perceptr(X, X_train, X_val, y_train, y_val, X_test,y_test):
    
    model = Perceptron()
    model.fit(X_train, y_train)

    sh=X.shape
    save_model(model,sh)

    tr_f1, val_f1, test_f1, tr_acc, val_acc,test_acc=model_performance(model, X_train, y_train, X_val, y_val, X_test, y_test)
    return tr_f1, val_f1, test_f1

#%%
def ridge(X, X_train, X_val, y_train, y_val, X_test,y_test):
    
    model = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1])
    model.fit(X_train, y_train)

    sh=X.shape
    save_model(model,sh)

    tr_f1, val_f1, test_f1, tr_acc, val_acc,test_acc=model_performance(model, X_train, y_train, X_val, y_val, X_test, y_test)
    return tr_f1, val_f1, test_f1
#%%
def naive_ba(X, X_train, X_val, y_train, y_val, X_test,y_test):

    model = GaussianNB()
    model.fit(X_train, y_train)
    sh=X.shape
    save_model(model,sh)

    tr_f1, val_f1, test_f1, tr_acc, val_acc,test_acc=model_performance(model, X_train, y_train, X_val, y_val, X_test, y_test)
    return tr_f1, val_f1, test_f1
#%%

def logregmodel(X, X_train, X_val, y_train, y_val, X_test,y_test):
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    sh=X.shape
    save_model(model,sh)

    tr_f1, val_f1, test_f1, tr_acc, val_acc,test_acc=model_performance(model, X_train, y_train, X_val, y_val, X_test, y_test)
    return tr_f1, val_f1, test_f1

#%%
def log_l1(X, X_train, X_val, y_train, y_val, X_test,y_test):

    model = SGDClassifier(loss="log", penalty="l1", random_state=42)
    model.fit(X_train, y_train)
    sh=X.shape
    save_model(model,sh)
    tr_f1, val_f1, test_f1, tr_acc, val_acc,test_acc=model_performance(model, X_train, y_train, X_val, y_val, X_test, y_test)
    return tr_f1, val_f1, test_f1




    

