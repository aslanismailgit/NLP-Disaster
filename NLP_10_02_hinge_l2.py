import numpy as np 
import pandas as pd 
#from sklearn import feature_extraction, linear_model, model_selection, preprocessing
import matplotlib.pyplot as plt
import time
from sklearn.feature_extraction.text import CountVectorizer
from NLP_functions import *

#%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

#%
#path = "/home/samsung-ub/Documents/Python/NLPDisaster/data/"
path="/Users/ismailaslan/Desktop/Python/NLPDisaster/data/"

train_df = pd.read_csv(path + "train.csv")
test_df = pd.read_csv(path + "test.csv")
t_df = pd.read_csv(path + "test_sonuc.csv")

bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
                                    token_pattern=r'\b\w+\b', min_df=1)

train_vectors_bi = bigram_vectorizer.fit_transform(train_df["text"])
test_vectors_bi = bigram_vectorizer.transform(test_df["text"])

X=train_vectors_bi.toarray()
y=train_df["target"]

X_test=test_vectors_bi.toarray()
y_test=t_df["target"]

X_train, X_val, y_train, y_val = train_test_split(X, y,test_size=0.15, random_state=42)
#%%
X_dummy, y_dummy, X_dummy_test, y_dummy_test = dummy()
X_dummy_train, X_dummy_val, y_dummy_train, y_dummy_val = train_test_split(
        X_dummy, y_dummy,test_size=0.15, random_state=42)

#%%
%clear
start_time=time.process_time()
alpha=[0.1, 0.01, 0.001, 0.0001]
#alp=0.05
path1 = "/Users/ismailaslan/Desktop/Python/NLPDisaster/my_models_10/"
for alp in alpha:
    print ("\n","="*40, "alpha = ", alp ,"="*40,"\n")
    model = SGDClassifier(
            alpha = alp, verbose=1,
            loss="hinge", early_stopping = True,
            max_iter = 200)
    model.fit(X_train, y_train)
    tr_f1, val_f1, test_f1, tr_acc, val_acc,test_acc = model_performance(
            model, X_train, y_train, X_val, y_val, X_test, y_test)
    save_model(model,path1)
    time_ = time.process_time()-start_time
    print ("\n","="*3,"Time for this iter %0.2f" % time_,"="*3,"\n")
    
Time_elapsed = time.process_time()-start_time
print ("\n","="*30,"Time elapsed %0.2f" % Time_elapsed,"="*30,"\n")


#%%
#np.ceil(10**6 / 6851) max iter
import pickle
path2 = "/Users/ismailaslan/Desktop/Python/NLPDisaster/my_models_10/"
filename = "SGDClassifier_17_145837.sav"
make_submittion(path2, filename, X_test)
model = pickle.load(open(path2 + filename, 'rb'))
tr_f1, val_f1, test_f1, tr_acc, val_acc,test_acc = model_performance(
            model, X_train, y_train, X_val, y_val, X_test, y_test)
#%% SGDClassifier_17_145837 0.77402







