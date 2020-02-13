import numpy as np 
import pandas as pd
import time
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

"""
6 different models are run. 
All models are almost overfit
need some regularization

los=["logReg","log","log","log","hinge","hinge"]
pen=["logReg","None", "l2", "l1", "l2", "l1"]



_log_l2_08_2239sav ======>
======== F1 SCORE =========
TR F1 0.996     VAL F1 0.762    TEST F1 0.7437
========= ACCURACY ============
TR ACC 0.996    VAL ACC 0.799   TEST ACC 0.787 **

_log_None_08_2238sav
======== F1 SCORE =========
TR F1 0.997     VAL F1 0.731    TEST F1 0.722 
========= ACCURACY ============
TR ACC 0.997    VAL ACC 0.770   TEST ACC 0.768 

_log_l1_08_2241sav ======>
======== F1 SCORE =========
TR F1 0.980     VAL F1 0.741    TEST F1 0.718 
========= ACCURACY ============
TR ACC 0.983    VAL ACC 0.778   TEST ACC 0.764 

_hinge_l1_08_2244sav ======>
======== F1 SCORE =========
TR F1 0.992     VAL F1 0.733    TEST F1 0.721 
========= ACCURACY ============
TR ACC 0.993    VAL ACC 0.769   TEST ACC 0.764 

_logReg_logReg_08_2237sav ======>
======== F1 SCORE =========
TR F1 0.993     VAL F1 0.763    TEST F1 0.737 
========= ACCURACY ============
TR ACC 0.994  VAL ACC 0.813     TEST ACC 0.794 **

_hinge_l2_08_2242sav ======>
======== F1 SCORE =========
TR F1 0.996     VAL F1 0.742    TEST F1 0.743 
========= ACCURACY ============
TR ACC 0.997    VAL ACC 0.783   TEST ACC 0.788 


filename="_logReg_logReg_31_1019.sav"      #81*
filename="_log_None_31_1027.sav"           #77
filename="_log_l2_31_1038.sav"              #79* 
filename="_log_l1_31_1055.sav"              # 77
filename="_hinge_l2_31_1107.sav"            # 78 *0.79345
filename="_hinge_l1_31_1125.sav"            #76*

"""

#%%
path="/Users/ismailaslan/Desktop/Python/NLPDisaster/data/"

train_df = pd.read_csv(path + "train.csv")
test_df = pd.read_csv(path + "test.csv")
#%%
t_df = pd.read_csv(path + "test_sonuc.csv")
y_test=t_df["target"]
#%%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

dim=train_df.shape[0]
count_vectorizer = feature_extraction.text.CountVectorizer()
train_vectors = count_vectorizer.fit_transform(train_df["text"][0:dim])
test_vectors = count_vectorizer.transform(test_df["text"][0:dim])
train_vectors_array=train_vectors.toarray()
test_vectors_array=test_vectors.toarray()

bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
                                    token_pattern=r'\b\w+\b', min_df=1)

train_vectors_bi = bigram_vectorizer.fit_transform(train_df["text"][0:dim])
test_vectors_bi = bigram_vectorizer.transform(test_df["text"][0:dim])

X=train_vectors_bi.toarray()
X_test=test_vectors_bi.toarray()
y=train_df["target"][0:dim]

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
        print("Train Error ==>  %0.3f" % train_Cls_Error,"Val Error ==>  %0.3f" % val_Cls_Error,"Test Error ==>  %0.3f" % test_Cls_Error)

    return Results, Resultsf

#%% ---------------- FUNCTIONS -----------------------
def save_model(model,los,pen,i):
    import time
    
    import pickle
    run_id = time.strftime("%d_%H%M.sav")
    path="/Users/ismailaslan/Desktop/Python/NLPDisaster/"
    filename=path+"my_models_02/" + "_" + los[i] + "_" + pen[i]+ "_"  + run_id
    # save the model to disk
    pickle.dump(model, open(filename, 'wb'))
    print(filename, "------->  saved")




