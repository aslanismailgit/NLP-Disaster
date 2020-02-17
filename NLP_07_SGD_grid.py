import numpy as np 
import pandas as pd
import time
from NLP_functions import *

from sklearn.feature_extraction.text import TfidfTransformer

#%%
path="/Users/ismailaslan/Desktop/Python/NLPDisaster/data/"

X_train, X_val, y_train, y_val, X_test, y_test = prep_data(path)
X=np.r_[X_train, X_val]
y=np.r_[y_train, y_val]

#reduction = TruncatedSVD(n_components=10000)

transformer = TfidfTransformer()
X_tfid=transformer.fit_transform(X).toarray()
X_test_tfid=transformer.transform(X_test).toarray()


#%%
dummy_clas1= np.random.normal(0, 1, 500).reshape(50,-1)
dummy_clas2=np.random.normal(1,1,500).reshape(50,-1)
X_dummy=np.concatenate((dummy_clas1, dummy_clas2))
X_test_dummy=np.concatenate(((np.random.normal(0, 1, 250).reshape(25,-1)),(np.random.normal(1, 1, 250).reshape(25,-1))))
y1=np.zeros((50,1))
y2=np.ones((50,1))
y_dummy=np.concatenate((y1,y2))
y_test_dummy=np.concatenate(((np.zeros((25,1))),(np.ones((25,1)))))
#%%
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

#%% ===== create model ===========
model = SGDClassifier(verbose=1)
parameters = {
            'loss': ("modified_huber", 'log', 'hinge'),
              'penalty': ['l1', 'l2', 'elasticnet'],
              'alpha': [0.001, 0.0001, 0.00001, 0.000001]
              }
scoring = ['accuracy', 'precision', "f1"]
grid_search = GridSearchCV(model, parameters, cv=12,scoring=scoring,refit='f1')
#grid_search.fit(X_dummy, y_dummy)
gridfunc(X,y)

#% ===== show best paramaters ==========
print(grid_search.best_score_)
best_params = grid_search.best_params_
print(best_params)

#% === save results ==========
path2="/Users/ismailaslan/Desktop/Python/NLPDisaster/my_models_07/"

res_df= pd.DataFrame(grid_search.cv_results_)
run_id = time.strftime("_%d_%H%M.csv")
res_df.to_csv(path2 + "_" + str(best_params["alpha"])+ "_" + best_params["loss"]+ "_" + best_params["penalty"] + run_id)

#% === calculate metrics ==========
y_pred = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1=f1_score(y_test, y_pred)
print(" ======== MODEL 1 COMPLETED ============")

gridfunc_tfid(X_tfid,y)
print(" ======== MODEL 2 COMPLETED ============")
#%%
def gridfunc(X,y):
    grid_search.fit(X, y)
    return grid_search
#%%
def gridfunc_tfid(X,y):
    grid_search_tfid.fit(X, y)
    return grid_search_tfid
