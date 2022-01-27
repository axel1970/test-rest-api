
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from spacy.lang.en.stop_words import STOP_WORDS
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import spacy

nlp = spacy.load('en_core_web_sm')
data_yelp = pd.read_csv('C:/Users/anike/OneDrive/Desktop/app/app/yelp_labelled.txt',sep='\t',header=None)
columan_name = ['Review', 'Sentiment']
data_yelp.columns = columan_name
data_amazon = pd.read_csv('C:/Users/anike/OneDrive/Desktop/app/app/amazon_cells_labelled.txt',sep='\t',header=None)
data_amazon.columns = columan_name
data_imdb = pd.read_csv('C:/Users/anike/OneDrive/Desktop/app/app/imdb_labelled.txt',sep='\t',header=None)
data_imdb.columns = columan_name
data = data_yelp.append([data_amazon, data_imdb],ignore_index=True)
print(data)
data['Sentiment'].value_counts()
# 1386 positive reviews
# 1362 Negative reviews
x = data['Review']
y = data['Sentiment']
import string
punct = string.punctuation
stopwords = list(STOP_WORDS) # list of stopwords

def text_data_cleaning(sentence):
  doc = nlp(sentence)

  tokens = [] # list of tokens
  for token in doc:
    if token.lemma_ != "-PRON-":
      temp = token.lemma_.lower().strip()
    else:
      temp = token.lower_
    tokens.append(temp)
 
  cleaned_tokens = []
  for token in tokens:
    if token not in stopwords and token not in punct:
      cleaned_tokens.append(token)
  return cleaned_tokens


tfidf = TfidfVectorizer(tokenizer=text_data_cleaning)
classifier = LinearSVC()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
clf = Pipeline([('tfidf',tfidf), ('clf',classifier)])
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy_score(y_test, y_pred)
print(clf.predict(["Wow, I am learning Natural Language P!"]))
clf.predict(["It's hard to learn new things!"])
print("the accuracy is : ",accuracy_score(y_test, y_pred))


pickle.dump(clf, open('sent.pkl', 'wb'))

     
