import numpy as np 
import pandas as pd
import time
from NLP_functions import *

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import time


#%%
#path="/Users/ismailaslan/Desktop/Python/NLPDisaster/data/"
path = "/home/samsung-ub/Documents/Python/NLPDisaster/data/"

from sklearn.feature_extraction.text import CountVectorizer
#    from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.model_selection import train_test_split

#    path="/Users/ismailaslan/Desktop/Python/NLPDisaster/data/"
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

#reduction = TruncatedSVD(n_components=10000)

#transformer = TfidfTransformer()
#X_tfid=transformer.fit_transform(X).toarray()
#X_test_tfid=transformer.transform(X_test).toarray()


#%%
dummy_clas1= np.random.normal(0, 1, 500).reshape(50,-1)
dummy_clas2=np.random.normal(1,1,500).reshape(50,-1)
X_dummy=np.concatenate((dummy_clas1, dummy_clas2))
X_test_dummy=np.concatenate(((np.random.normal(0, 1, 250).reshape(25,-1)),(np.random.normal(1, 1, 250).reshape(25,-1))))
y1=np.zeros((50,1))
y2=np.ones((50,1))
y_dummy=np.concatenate((y1,y2))
y_test_dummy=np.concatenate(((np.zeros((25,1))),(np.ones((25,1)))))

#%% ---------------- polynomial ------------------
from sklearn.svm import SVC
poly_kernel_svm_clf = Pipeline([
("transformer", TfidfTransformer()),
("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5, verbose=1))
])
#%% 
start_time=time.process_time()
poly_kernel_svm_clf.fit(X, y)
#poly_kernel_svm_clf.fit(X, y)
print("time elapsed =============== %0.2f" % (time.process_time()-start_time))
y_pred=poly_kernel_svm_clf.predict(X_test)
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
f1=f1_score(y_pred,y_test)
print("f1 ------ %0.3f" % f1)
acc1=accuracy_score(y_pred,y_test)
print ("Accuracy ---- {:.3f}" .format(acc1))
#%% ------------- RBF ------------
from sklearn.svm import SVC
rbf_kernel_svm_clf = Pipeline([
("transformer", TfidfTransformer()),
("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001, verbose=1))
])


#%% 
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

start_time=time.process_time()
#rbf_kernel_svm_clf.fit(X_dummy, y_dummy)
rbf_kernel_svm_clf.fit(X, y)
print("time elapsed =============== %0.2f" % (time.process_time()-start_time))
#y_pred=rbf_kernel_svm_clf.predict(X_test_dummy)
y_pred=rbf_kernel_svm_clf.predict(X_test)
f1=f1_score(y_pred,y_test)
#f1=f1_score(y_pred,y_test_dummy)
print("f1 ------ %0.3f" % f1)
#acc1=accuracy_score(y_pred,y_test_dummy)
acc1=accuracy_score(y_pred,y_test)
print ("Accuracy ---- {:.3f}" .format(acc1))

#%%
