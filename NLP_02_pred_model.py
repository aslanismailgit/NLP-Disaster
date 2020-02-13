import numpy as np 
import pandas as pd
import time
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
"""
this model uses the classification result of again below models
to predict test data. So it is a model of model. the problem is what we will
use as test-X. I  take the outputs of each model, 
I mean X-test has 6 columns.(as it should be). in any case, if you dont
train the "entire model" it gives no good resutls.
use ensemle instead

"""
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


#%%

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score


#%%
X_train, X_val, y_train, y_val = train_test_split(X, y,random_state=42)
#%%
data={'LogReg': [0], 
     'SGD_log_None': [0],
     'SGD_log_l2': [0],
     'SGD_log_l1': [0],
     'SGD_hinge_l2': [0],
     'SGD_hinge_l1': [0]}

ind =['train_Cls_Error', 'val_Cls_Error', "test_Cls_Error"]
indf =['train_f1', 'val_f1', "test_"]
Results = pd.DataFrame(data, index =ind) 
Resultsf = pd.DataFrame(data, index =indf) 
#%%
los=["logReg","log","log","log","hinge","hinge"]
pen=["logReg","None", "l2", "l1", "l2", "l1"]

#%%
start_time=time.process_time()

Results, Resultsf = runDifferentModels(X_train,y_train,X_val,y_val,los,pen,Results,Resultsf, X_test) 
   
print("--- %0.2f seconds ---" % (time.process_time() - start_time))
#%% ---------------- FUNCTIONS -----------------------
def runDifferentModels(X_train,y_train,X_val,y_val,los,pen, Results, Resultsf, X_test):
    for i in range(len(los)):
        if i==0:
            model = LogisticRegression()
            model.fit(X_train, y_train)
            
        else:
            model = SGDClassifier(loss=los[i], penalty=pen[i], random_state=42)
            model.fit(X_train, y_train)
        save_model(model,los,pen,i)

        y_train_predict = model.predict(X_train)
        train_Cls_Error=1 - np.mean(y_train == y_train_predict)
        tr_f1=f1_score(y_train , y_train_predict)

        y_val_predict = model.predict(X_val)
        val_Cls_Error=1 - np.mean(y_val == y_val_predict)
        val_f1=f1_score(y_val, y_val_predict)
            
        y_test_predict = model.predict(X_test)
        test_Cls_Error=1 - np.mean(y_test == y_test_predict)
        test_f1=f1_score(y_test, y_test_predict)
        
        Results.iloc[:,i]=train_Cls_Error,val_Cls_Error,test_Cls_Error
        Resultsf.iloc[:,i]=tr_f1, val_f1, test_f1
        print("i=",i," ", los[i]," ", pen[i])
        print("================================")
        print("--- %0.2f seconds passed---" % (time.process_time() - start_time))
        print("Train Error ==>",train_Cls_Error,"Val Error ==>",val_Cls_Error,"Test Error ==>",test_Cls_Error)
    return Results, Resultsf

#%% ---------------- FUNCTIONS -----------------------
def save_model(model,los,pen,i):
    import time
    
    import pickle
    run_id = time.strftime("%d_%H%M.sav")
    path2="/Users/ismailaslan/Desktop/Python/NLPDisaster/my_models_02_pred_model/"
    filename=path2+ "_" + los[i] + "_" + pen[i]+ "_"  + run_id
    # save the model to disk
    pickle.dump(model, open(filename, 'wb'))
    print(filename, "------->  saved")




