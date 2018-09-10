import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import string
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score

pd.options.display.max_columns = 100

from matplotlib import pyplot as plt
import numpy as np

import seaborn as sns

import pylab as plot


def process_text(text):
    '''
    What will be covered:
    1. Remove punctuation
    2. Remove stopwords
    3. Return list of clean text words
    '''

    # 1
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    # 2
    clean_words = [PorterStemmer().stem(word) for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    # LancasterStemmer().stem(f)
    # SnowballStemmer('english').stem(f)
    # 3
    return clean_words



dataset = pd.read_csv('spam.csv',encoding='latin-1')
dataset.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
dataset = dataset.rename(columns={'v1': 'class','v2': 'text'})
dataset['length'] = dataset['text'].apply(len)




#split training set and testing set
msg_train, msg_test, class_train, class_test = train_test_split(dataset['text'],dataset['class'],test_size=0.2)
pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=process_text,ngram_range=(1,2))), # converts strings to integer counts
    ('tfidf',TfidfTransformer()), # converts integer counts to weighted TF-IDF scores
    ('classifier',MultinomialNB(alpha=0.1)) # train on TF-IDF vectors with Naive Bayes classifier
])



pipeline.fit(msg_train,class_train)
predictions = pipeline.predict(msg_test)
print(classification_report(class_test,predictions))
cm=confusion_matrix(class_test,predictions,labels=['spam','ham'])
print("accuracy :",accuracy_score(class_test,predictions))
print(cm)
submission = pd.DataFrame({
        "Classes": predictions
    })
submission.to_csv('results.csv', index=True)

