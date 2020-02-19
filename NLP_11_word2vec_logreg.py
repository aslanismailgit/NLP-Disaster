import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import time

from NLP_functions import *

#%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#%%
path = "/home/samsung-ub/Documents/Python/NLPDisaster/data/"
#path="/Users/ismailaslan/Desktop/Python/NLPDisaster/data/"

train_df = pd.read_csv(path + "train.csv")
test_df = pd.read_csv(path + "test.csv")
t_df = pd.read_csv(path + "test_sonuc.csv")

tweet = train_df["text"]
y=train_df["target"]

test_tweet=test_df["text"]
y_test=t_df["target"]

#%%
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

train_df["tokens"] = train_df["text"].apply(tokenizer.tokenize)
train_df.head()
test_df["tokens"] = test_df["text"].apply(tokenizer.tokenize)
test_df.head()

#%%

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

all_words = [word for tokens in train_df["tokens"] for word in tokens]
sentence_lengths = [len(tokens) for tokens in train_df["tokens"]]
VOCAB = sorted(list(set(all_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
print("Max sentence length is %s" % max(sentence_lengths))


#%%
fig = plt.figure(figsize=(10, 10)) 
plt.xlabel('Sentence length')
plt.ylabel('Number of sentences')
plt.hist(sentence_lengths)
plt.show()

#%%
import gensim
path = "/home/samsung-ub/Documents/Python/Datasets/"
word2vec_path = "GoogleNews-vectors-negative300.bin"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(path+word2vec_path, binary=True)

#%%:
def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, train_df, generate_missing=False):
    embeddings = train_df['tokens'].apply(lambda x: get_average_word2vec(x, vectors, 
                                                                                generate_missing=generate_missing))
    return list(embeddings)
#%%
embeddings = get_word2vec_embeddings(word2vec, train_df)
X_train_word2vec, X_val_word2vec, y_test_word2vec, y_val_word2vec = train_test_split(embeddings, y,
                                                                                       test_size=0.2, random_state=40)
#%%
X_test_word2vec = get_word2vec_embeddings(word2vec, test_df)
          
#%%
#clf_w2v = LogisticRegression(C=30.0, solver='newton-cg')

model = LogisticRegression(C=5.0, class_weight='balanced', solver='newton-cg', 
                         multi_class='multinomial', n_jobs=-1, random_state=40)
model.fit(X_train_word2vec, y_train_word2vec)
tr_f1, val_f1, test_f1, tr_acc, val_acc,test_acc, test_f1_weighted,test_f1_macro,test_f1_micro, test_f1_None = model_performance(
        model, X_train_word2vec, y_train_word2vec, X_val_word2vec, y_val_word2vec, X_test_word2vec, y_test)

#%%
%clear
start_time=time.process_time()
cc=[0.1, 1, 5, 8, 10, 12, 16, 20]
C_values, test_f1_weighteds,test_f1_macros,test_f1_micros, test_f1_Nones=[],[],[],[],[]

#path1 = "/Users/ismailaslan/Desktop/Python/NLPDisaster/my_models_04_02/"
path1 = "/home/samsung-ub/Documents/Python/NLPDisaster/my_models_11/"
for c in cc:
    print ("\n","="*40, "C = ", c ,"="*40,"\n")
    model = LogisticRegression(C=c, verbose=0, solver="lbfgs", max_iter=150)
    model.fit(X_train_word2vec, y_train_word2vec)
    tr_f1, val_f1, test_f1, tr_acc, val_acc,test_acc, test_f1_weighted,test_f1_macro,test_f1_micro, test_f1_None = model_performance(
            model, X_train_word2vec, y_train_word2vec, X_val_word2vec, y_val_word2vec, X_test_word2vec, y_test)
    save_model(model,path1)

    C_values.append(c)
    test_f1_weighteds.append(test_f1_weighted)
    test_f1_macros.append(test_f1_macro)
    test_f1_micros.append(test_f1_micro)
    test_f1_Nones.append(test_f1_None)

Time_elapsed = time.process_time()-start_time
print ("\n","="*30,"Time elapsed %0.2f" % Time_elapsed,"="*30,"\n")


#%%
plt.plot(C_values,test_f1_weighteds,"b.")
plt.plot(C_values,test_f1_macros,"r.")
plt.plot(C_values,test_f1_micros,"c.")
plt.show()
#%%
#np.ceil(10**6 / 6851) max iter
import pickle
path1 = "/home/samsung-ub/Documents/Python/NLPDisaster/data/"
#path2 = "/Users/ismailaslan/Desktop/Python/NLPDisaster/my_models_10/"
path2 = "/home/samsung-ub/Documents/Python/NLPDisaster/my_models_11/"

filename = "LogisticRegression_19_104412.sav"
make_submission(path1, path2, filename, X_test_word2vec)

#%% 
LogisticRegression_19_102648.sav 0.78016
LogisticRegression_19_104412     0.78732







