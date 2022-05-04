#The Hotel Review Dataset was downloaded from Kaggle
#This dataset originally contains a total of 10,000 entries of customers' reviews on different hotels that they stayed, however, only 5 points or 1 point of reviews were kept for analysis
#The purpose of this NLP project is to perform a sentiment analysis to decide if a review is positive (rated as 5 points) or negative (rated as 1 point)

import pandas as pd
import numpy as np
import string

#read in data and select all the data that showed 5 points or 1 point of rating
data = pd.read_csv('FILE PATH')
df = data[(data['reviews.rating']==1) | (data['reviews.rating']==5)]

#reset index for the new dataset df
df = df.reset_index().drop('index', axis=1)

#data cleaning--removed punctuation, standardized all the texts to lowercase, removed stop words and completed stemming
review = df['reviews.text'].str.lower()
review = review.str.translate(str.maketrans('', '', string.punctuation))
review = review.str.split()
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
sw = stopwords.words()
sw = set(sw)
corpus = []
for i in range(len(review)):
	words = review[i]
	words = [ps.stem(word) for word in words if not word in sw]
	words = ' '.join(words)
	corpus.append(words)

#transform the data
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x= cv.fit_transform(corpus).toarray()
y = df['reviews.rating'].values

#perform train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=13)

#train a model using GuassianNB as a classifier
from sklearn.naive_bayes import GaussianNB
gb = GaussianNB()
gb.fit(x_train, y_train)
pred = gb.predict(x_test)

#evaluate the model using testing data
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

#train a model using MultinomialNB as a classifier
from sklearn.naive_bayes import MultinomialNB
mb = MultinomialNB()
mb.fit(x_train, y_train)
predict = mb.predict(x_test)

#evaluate the model using testing data
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predict))
print(confusion_matrix(y_test, predict))

#results from the MultinomialNB showed a much improved accuracy (0.97). 
