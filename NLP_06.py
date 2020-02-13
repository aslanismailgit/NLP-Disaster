"""
savd models need to be reported

Classisfiers:
Random forest
naive bayes
ridge
perceptron
log regression
log l1


results @:
/Users/ismailaslan/Desktop/Python/NLPDisaster/my_models_06
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
path="/Users/ismailaslan/Desktop/Python/NLPDisaster"

train_df = pd.read_csv(path+"/data/train.csv")
test_df = pd.read_csv(path+"/data/test.csv")
#%%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

dim=train_df.shape[0]

bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
                                    token_pattern=r'\b\w+\b', min_df=1)

train_vectors_bi = bigram_vectorizer.fit_transform(train_df["text"][0:dim])
test_vectors_bi = bigram_vectorizer.transform(test_df["text"][0:dim])

X=train_vectors_bi.toarray()
test_vectors_bi=test_vectors_bi.toarray()
y=train_df["target"][0:dim]

X_train, X_val, y_train, y_val = train_test_split(X, y,random_state=42)


#%%
#transformer = TfidfTransformer()
#transformer.fit_transform(counts).toarray()

#%%
#path="/home/samsung-ub/Documents/Python/NLPDisaster"

t_df = pd.read_csv(path+"/data/test_sonuc.csv")
y_test=t_df["target"]



#%%
start_time=time.process_time()

tr_f1, val_f1, test_f1 = log_l1(X,y, test_vectors_bi,y_test)
tr_f1l, val_f1l, test_f1l = logregmodel(X,y, test_vectors_bi,y_test)
tr_f1nb, val_f1nb, test_f1nb = naive_ba(X,y, test_vectors_bi,y_test)
tr_f1rd, val_f1rd, test_f1rd = ridge(X,y, test_vectors_bi,y_test)
tr_f1pr, val_f1pr, test_f1pr = perceptr(X,y, test_vectors_bi,y_test)
tr_f1rf, val_f1rf, test_f1rf = rand_forest(X,y, test_vectors_bi,y_test)

print("--- %0.2f seconds ---" % (time.process_time() - start_time))

#%%
def rand_forest(X_train, X_val, y_train, y_val, X_test,y_test):
    
    model = RandomForestClassifier(max_depth=2, random_state=0)
    model.fit(X_train, y_train)

    sh=Xm.shape
    save_model(model,sh)

    tr_f1, val_f1, test_f1=model_performance(model, X_train, y_train, X_val, y_val, X_test, y_test)
    return tr_f1, val_f1, test_f1

#%%
def perceptr(X_train, X_val, y_train, y_val, X_test,y_test):
    
    model = Perceptron()
    model.fit(X_train, y_train)

    sh=Xm.shape
    save_model(model,sh)

    tr_f1, val_f1, test_f1=model_performance(model, X_train, y_train, X_val, y_val, X_test, y_test)
    return tr_f1, val_f1, test_f1

#%%
def ridge(X_train, X_val, y_train, y_val, X_test,y_test):
    
    model = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1])
    model.fit(X_train, y_train)

    sh=Xm.shape
    save_model(model,sh)

    tr_f1, val_f1, test_f1=model_performance(model, X_train, y_train, X_val, y_val, X_test, y_test)
    return tr_f1, val_f1, test_f1
#%%
def naive_ba(X_train, X_val, y_train, y_val, X_test,y_test):

    model = GaussianNB()
    model.fit(X_train, y_train)
    sh=Xm.shape
    save_model(model,sh)

    tr_f1, val_f1, test_f1=model_performance(model, X_train, y_train, X_val, y_val, X_test, y_test)
    return tr_f1, val_f1, test_f1
#%%

def logregmodel(X_train, X_val, y_train, y_val, X_test,y_test):
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    sh=Xm.shape
    save_model(model,sh)

    tr_f1, val_f1, test_f1=model_performance(model, X_train, y_train, X_val, y_val, X_test, y_test)
    return tr_f1, val_f1, test_f1

#%%
def log_l1(X_train, X_val, y_train, y_val, X_test,y_test):

    model = SGDClassifier(loss="log", penalty="l1", random_state=42)
    model.fit(X_train, y_train)
    sh=Xm.shape
    save_model(model,sh)
    tr_f1, val_f1, test_f1=model_performance(model, X_train, y_train, X_val, y_val, X_test, y_test)
    return tr_f1, val_f1, test_f1



    
