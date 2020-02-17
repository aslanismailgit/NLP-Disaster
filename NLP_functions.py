
"""
Created on Sun Feb  9 22:43:48 2020

@author: ismailaslan
"""
import numpy as np 
import pandas as pd
import time

#%%
def prep_data(path):
    from sklearn.feature_extraction.text import CountVectorizer
#    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.model_selection import train_test_split
    
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
    
    X_train, X_val, y_train, y_val = train_test_split(X, y,random_state=42)
    
    return X_train, X_val, y_train, y_val, X_test, y_test
#%%
def metafeatures(train_df ,test_df,tr_meta, test_meta ):
    from wordcloud import STOPWORDS
    import string


    # word_count
    tr_meta['word_count'] = train_df['text'].apply(lambda x: len(str(x).split()))
    test_meta['word_count'] = test_df['text'].apply(lambda x: len(str(x).split()))
    
    # unique_word_count
    tr_meta['unique_word_count'] = train_df['text'].apply(lambda x: len(set(str(x).split())))
    test_meta['unique_word_count'] = test_df['text'].apply(lambda x: len(set(str(x).split())))
    
    # stop_word_count
    tr_meta['stop_word_count'] = train_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
    test_meta['stop_word_count'] = test_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
    
    # url_count
    tr_meta['url_count'] = train_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))
    test_meta['url_count'] = test_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))
    
    # mean_word_length
    tr_meta['mean_word_length'] = train_df['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    test_meta['mean_word_length'] = test_df['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    
    # char_count
    tr_meta['char_count'] = train_df['text'].apply(lambda x: len(str(x)))
    test_meta['char_count'] = test_df['text'].apply(lambda x: len(str(x)))
    
    # punctuation_count
    tr_meta['punctuation_count'] = train_df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    test_meta['punctuation_count'] = test_df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    
    # hashtag_count
    tr_meta['hashtag_count'] = train_df['text'].apply(lambda x: len([c for c in str(x) if c == '#']))
    test_meta['hashtag_count'] = test_df['text'].apply(lambda x: len([c for c in str(x) if c == '#']))
    
    # mention_count
    tr_meta['mention_count'] = train_df['text'].apply(lambda x: len([c for c in str(x) if c == '@']))
    test_meta['mention_count'] = test_df['text'].apply(lambda x: len([c for c in str(x) if c == '@']))

    return tr_meta, test_meta

#%% ---------------- FUNCTIONS -----------------------
def save_model(model, path):
    import time
    import pickle
    model_name = type(model).__name__
    run_id = time.strftime("_%d_%H%M%S.sav")
#    path="/Users/ismailaslan/Desktop/Python/NLPDisaster/"
    filename=path + model_name + run_id
    # save the model to disk
    pickle.dump(model, open(filename, 'wb'))
    
    print ("\nSaving model .......")
    print(filename, "------->  saved")

#%%
def dummy():
    dummy_clas1= np.random.normal(0, 1, 500).reshape(50,-1)
    dummy_clas2=np.random.normal(1,1,500).reshape(50,-1)
    X_dummy=np.concatenate((dummy_clas1, dummy_clas2))
    X_dummy_test=np.concatenate(((np.random.normal(0, 1, 250).reshape(25,-1)),(np.random.normal(1, 1, 250).reshape(25,-1))))
    y1=np.zeros((50,1)).astype("int")
    y2=np.ones((50,1)).astype("int")
    y_dummy=np.concatenate((y1,y2))
    y_dummy_test=np.concatenate(((np.zeros((25,1))),(np.ones((25,1))))).astype("int")
    
    return X_dummy, y_dummy, X_dummy_test, y_dummy_test

#%%
def model_performance(model, X_train, y_train, X_val, y_val, X_test, y_test):
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score

    y_train_predict = model.predict(X_train)
    tr_f1=f1_score(y_train, y_train_predict, average='weighted')
    tr_acc=accuracy_score(y_train, y_train_predict)

    y_val_predict = model.predict(X_val)
    val_f1=f1_score(y_val, y_val_predict, average='weighted')
    val_acc=accuracy_score(y_val, y_val_predict)

        
    y_test_predict = model.predict(X_test)
    test_f1=f1_score(y_test, y_test_predict, average='weighted')
    test_acc=accuracy_score(y_test, y_test_predict)

    test_f1_weighted=f1_score(y_test, y_test_predict, average='weighted')
    test_f1_macro=f1_score(y_test, y_test_predict, average='macro')
    test_f1_micro=f1_score(y_test, y_test_predict, average='micro')
    test_f1_None=f1_score(y_test, y_test_predict, average=None)

    print("\n", "="*30,"   Evaluate Model   ", "="*30, "\n")
    print()
    print("Train Error - Acc ==>  %0.3f" % tr_acc,"Val Error - Acc ==>  %0.3f" % val_acc,"Test Error - Acc ==>  %0.3f" % test_acc)
    print("Train Error - F1  ==>  %0.3f" % tr_f1, "Val Error - F1  ==>  %0.3f" % val_f1, "Test Error - F1  ==>  %0.3f" % test_f1)
    print("Test F1 Error - weighted ==>  %0.3f" % test_f1_weighted)
    print("Test F1 Error - macro    ==>  %0.3f" % test_f1_macro)
    print("Test F1 Error - micro    ==>  %0.3f" % test_f1_micro)
    print("Test F1 Error - none     ==>  %0.3f" % test_f1_None[0])
    
    return tr_f1, val_f1, test_f1, tr_acc, val_acc,test_acc, test_f1_weighted,test_f1_macro,test_f1_micro, test_f1_None

#%%
def make_submission(path2, filename, X_test):
    import pickle
    path1 = "/Users/ismailaslan/Desktop/Python/NLPDisaster/data/"
#    path2 = "/Users/ismailaslan/Desktop/Python/NLPDisaster/my_models_04_02/"
    sample_submission = pd.read_csv(path1 + "samplesubmission.csv")
#    filename="LogisticRegression_16_153913.sav"
    model = pickle.load(open(path2 + filename, 'rb'))
    sample_submission["target"] = model.predict(X_test)
    sample_submission.head()
    sample_submission.to_csv(path2 + "submission.csv", index=False)

#%%
#def submit(model, X_test,path):
#    sample_submission = pd.read_csv("samplesubmission.csv")
#    sample_submission["target"] = model.predict(X_test)
#    sample_submission.head()
#    sample_submission.to_csv("submission.csv", index=False)

#import pickle
#path1 = "/Users/ismailaslan/Desktop/Python/NLPDisaster/data/"
#path2 = "/Users/ismailaslan/Desktop/Python/NLPDisaster/my_models_04_02/"
#sample_submission = pd.read_csv(path1 + "samplesubmission.csv")
#filename="LogisticRegression_16_153913.sav"
#model = pickle.load(open(path2 + filename, 'rb'))
#sample_submission["target"] = model.predict(X_test)
#sample_submission.head()
#sample_submission.to_csv(path2 + "submission.csv", index=False)

#%%
def getSavedModelPerformance(model):
    import pickle
    #pickle.dump(model, open(filename, 'wb'))

    filename="7613_91660_08_0004.sav"
    model = pickle.load(open(path+"/my_models/"+filename, 'rb'))
    tr_f1, val_f1, test_f1, tr_acc, val_acc,test_acc = model_performance(model, X_train, y_train, X_val, y_val, X_test, y_test)
    return tr_f1, val_f1, test_f1, tr_acc, val_acc,test_acc

#result = model.score(X_val, y_val)
    
#%%
def readFilesinFolder(path):
    import os
#    path = '/Users/ismailaslan/Desktop/Python/NLPDisaster/my_models_02'
    
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.sav' in file:
                files.append(os.path.join(r, file))
#    
#    for f in files:
#        print(f)
    return files

#%%
"""
numRows=X.shape[0]
numCols=len(files)
Y= pd.DataFrame(index=range(numRows),columns=range(numCols))
path = '/Users/ismailaslan/Desktop/Python/NLPDisaster/my_models_02'
nn=0
for model in files:
    name = type(model).__name__
    
    loaded_model = pickle.load(open(model, 'rb'))
    y_pred = loaded_model.predict(X)
    
    Y=Y.rename({nn: name}, axis='columns')
    Y[name]=y_pred
    nn+=1
    print("model ----->  %s ------ pred complete ------" % name)

Y["target"]=y
Y.to_csv(path + name + "/model_preds.csv",index = None)

#%%
        scores = {
            'model': str(model),
            'name': name,
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'time': [],
        }


            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            scores['time'].append(time.time() - start)
            scores['accuracy'].append(accuracy_score(y_test, y_pred))
            scores['precision'].append(precision_score(y_test, y_pred, average='weighted'))
            scores['recall'].append(recall_score(y_test, y_pred, average='weighted'))
            scores['f1'].append(f1_score(y_test, y_pred, average='weighted'))

"""