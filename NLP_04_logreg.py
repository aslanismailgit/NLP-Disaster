"""
This is log reg over different size of feature and sample sets

needs to be rerun to collect results

- nice 2 plots:
    number of features less than X occurance 
    number of samples greater than x occurance 

needs to be rerun to collect results

"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
import matplotlib.pyplot as plt
import time
#%%
train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")
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

#%%
#transformer = TfidfTransformer()
#transformer.fit_transform(counts).toarray()

#%%
sx=sum(X)
ind=[]
for i in range(0,31):
#    print(i)
    ind_=np.where(sx>i)
    ind.append(len(ind_[0]))
    print("more than %s instance ===" % i)
    print(len(ind_[0]))

plt.plot(ind)
plt.ylabel("Number of features longer than x")
plt.xlabel("Number of values in a feature")
len(len(sx)-(np.where(sx>6))[0])

#%%
sxc=sum(np.transpose(X))
ind=[]
for i in range(0,max(sxc)+1):
#    print(i)
    ind_=np.where(sxc>i)
    ind.append(len(ind_[0]))
    print("more than %s instance ===>" % i)
    print(len(ind_[0]))

plt.plot(ind)
plt.ylabel("Number of instances longer than x")
plt.xlabel("Number of values in an instance")
len(len(sx)-(np.where(sxc>3))[0])

#%%
t_df = pd.read_csv("./data/test_sonuc.csv")
y_test=t_df["target"]

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

#%%
tr_f1, val_f1, test_f1 = log_l1(X,y, test_vectors_bi,y_test)
tr_f1l, val_f1l, test_f1l = logregmodel(X,y, test_vectors_bi,y_test)

#%%
sx=sum(X)
sxc=sum(np.transpose(X))
start_time=time.process_time()
ff1=2
ff2=15
ins1=2
ins2=10

a=(ff2-ff1,ins2-ins1)
Results_train=np.zeros(a)
Results_val=np.zeros(a)
Results_test=np.zeros(a)

Results_train_log=np.zeros(a)
Results_val_log=np.zeros(a)
Results_test_log=np.zeros(a)
for ff in range(ff1,ff2):
    for ins in range (ins1,ins2):
        
        ind=np.where(sx>ff)
        Xm=X[:,ind[0]]
        X_test=test_vectors_bi[:,ind[0]]

        ind=np.where(sxc>ins)
        Xm=Xm[ind[0]]
        ym=y[ind[0]]
        print("X shape =====>", Xm.shape)
        print("ym shape ====>", ym.shape)
        print("X_test shape ========>", X_test.shape)
        print("y_test shape ========>", y_test.shape)
        
#        transformer = TfidfTransformer()
#        Xm=transformer.fit_transform(Xm).toarray()
#        X_test==transformer.transform(X_test).toarray()
        
        
        tr_f1, val_f1, test_f1 = log_l1(X,y, X_test,y_test)
        
        Results_train[ff-ff1,ins-ins1]=tr_f1
        Results_val[ff-ff1,ins-ins1]=val_f1
        Results_test[ff-ff1,ins-ins1]=test_f1
        
        
        print("--- %0.2f seconds passed---" % (time.process_time() - start_time))
        print("Train Error ======>",tr_f1)
        print("Val Error ========>",val_f1)
        print("Test Error ========>",test_f1)
        
        tr_f1, val_f1, test_f1 = logregmodel(Xm,ym, X_test,y_test)
        
        Results_train_log[ff-ff1,ins-ins1]=tr_f1
        Results_val_log[ff-ff1,ins-ins1]=val_f1
        Results_test_log[ff-ff1,ins-ins1]=test_f1
        
        
        print("--- %0.2f seconds passed---" % (time.process_time() - start_time))
        print("Train Error ======>",tr_f1)
        print("Val Error ========>",val_f1)
        print("Test Error ========>",test_f1)
#        
#        print("Test Error =======>",test_Cls_Error)
#        print("Test score========>", model.score(X_test, y_test))


#%%

def logregmodel(Xm,ym, X_test,y_test):
    start_time=time.process_time()
    X_train, X_val, y_train, y_val = train_test_split(Xm, ym,random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    sh=Xm.shape
    save_model(model,sh)

    y_train_predict = model.predict(X_train)
    tr_f1=f1_score(y_train, y_train_predict)
    
    y_val_predict = model.predict(X_val)
    val_f1=f1_score(y_val, y_val_predict)
        
    y_test_predict = model.predict(X_test)
    test_f1=f1_score(y_test, y_test_predict)

    return tr_f1, val_f1, test_f1

#%%
def log_l1(Xm,ym, X_test,y_test):
    start_time=time.process_time()
    X_train, X_val, y_train, y_val = train_test_split(Xm, ym,random_state=42)

    model = SGDClassifier(loss="log", penalty="l1", random_state=42)
    model.fit(X_train, y_train)
    sh=Xm.shape
    save_model(model,sh)

    y_train_predict = model.predict(X_train)
    tr_f1=f1_score(y_train, y_train_predict)
    
    y_val_predict = model.predict(X_val)
    val_f1=f1_score(y_val, y_val_predict)
        
    y_test_predict = model.predict(X_test)
    test_f1=f1_score(y_test, y_test_predict)

    return tr_f1, val_f1, test_f1



