
"""
Calssifiers run over different size of train/val k-fold

Classifiers:
= [
    ("SGD", SGDClassifier(max_iter=100)),
    ("ASGD", SGDClassifier(average=True)),
    ("Perceptron", Perceptron()),
    ("Log regression", PassiveAggressiveClassifier(loss='log',
                                                         C=1.0, tol=1e-4)),
    ("Passive-Aggressive I", PassiveAggressiveClassifier(loss='hinge',
                                                         C=1.0, tol=1e-4)),
    ("Passive-Aggressive II", PassiveAggressiveClassifier(loss='squared_hinge',
                                                          C=1.0, tol=1e-4)),
    ("SAG", LogisticRegression(solver='sag', tol=1e-1, C=1.e4 / X.shape[0]))

without tfid

results@
/Users/ismailaslan/Desktop/Python/NLPDisaster/my_models_03
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction #, linear_model, model_selection, preprocessing
from sklearn.feature_extraction.text import CountVectorizer

#%%
train_df = pd.read_csv("/Users/ismailaslan/Run_mac/data/train.csv")
test_df = pd.read_csv("/Users/ismailaslan/Run_mac/data/test.csv")
#%%

#from sklearn.feature_extraction.text import TfidfTransformer

dim=train_df.shape[0]
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
                                    token_pattern=r'\b\w+\b', min_df=1)

train_vectors_bi = bigram_vectorizer.fit_transform(train_df["text"][0:dim])
test_vectors_bi = bigram_vectorizer.transform(test_df["text"][0:dim])

X=train_vectors_bi.toarray()
test_vectors_bi=test_vectors_bi.toarray()
y=train_df["target"][0:dim]

#%%
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression

#%%
heldout = [ 0.90, 0.80, 0.75] 
#heldout = [0.80]
rounds = 3

data={'SGD': [0], 
     'ASGD': [0],
     'Perceptron': [0],
     'Log_Reg': [0],
     'PasAgg_I': [0],
     'PasAgg_II': [0],
     'SAG': [0],
     }
Results = pd.DataFrame(data, index =heldout) 

#%%
start_time=time.process_time()
classifiers = [
    ("SGD", SGDClassifier(max_iter=100)),
    ("ASGD", SGDClassifier(average=True)),
    ("Perceptron", Perceptron()),
    ("Log regression", PassiveAggressiveClassifier(loss='log',
                                                         C=1.0, tol=1e-4)),
    ("Passive-Aggressive I", PassiveAggressiveClassifier(loss='hinge',
                                                         C=1.0, tol=1e-4)),
    ("Passive-Aggressive II", PassiveAggressiveClassifier(loss='squared_hinge',
                                                          C=1.0, tol=1e-4)),
    ("SAG", LogisticRegression(solver='sag', tol=1e-1, C=1.e4 / X.shape[0]))
]

xx = 1. - np.array(heldout)
j=0
for name, clf in classifiers:
    print("--- %0.2f seconds ---" % (time.process_time() - start_time))
    print("name",name)
    print("training %s" % name)
#    rng = np.random.RandomState(42)
    yy = []
    for i in heldout:
        yy_ = []
        for r in range(rounds):
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=i)#, random_state=rng)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            yy_.append(1 - np.mean(y_pred == y_test))
            print("round=", r,"mean error", name, 1 - np.mean(y_pred == y_test))
            save_model(clf,name,i,r)
        yy.append(np.mean(yy_))
    Results.iloc[:,j]=yy
    j+=1
    plt.plot(xx, yy,"x--", label=name)

plt.legend(loc="upper right")
plt.xlabel("Proportion train")
plt.ylabel("Test Error Rate")
plt.show()
plt.savefig("Results.png")

print("--- %0.2f seconds ---" % (time.process_time() - start_time))

