import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('FILE PATH')

#basic information about the dataset
data.info()
data.head()
data.columns
data.index
data.isnull().sum()

#drop the two columns that contain large portion of missing data
df = data.drop(['reviews.userCity', 'reviews.userProvince', 'id'], axis=1)
df['reviews.text'].iloc[0]
df['reviews.text'].iloc[1]

#data cleaning and preprocessing
#remove punctuation
#standardize text to lower case
#tokenize text
#remove stopwords and stem text

import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()
sw = stopwords.words('english')
sw = set(sw)

df['review'] = df['reviews.text'].str.translate(str.maketrans('','', string.punctuation))
df['review'] = df['review'].str.lower()

from nltk.tokenize import word_tokenize
df['review'] = [nltk.word_tokenize(str(comment)) for comment in df['review']]

corpus = []
review = df['review']
len(review)
for i in range(len(review)):
	words = review[i]
	words = [ps.stem(word) for word in words if word not in sw]
	words = ' '.join(words)
	corpus.append(words)

#check the result
corpus[3]
corpus[2]

#create a column that shows whether the review is positive, negative or neutral
from textblob import TextBlob
df['review'] = corpus
def subjectivity(text):
	return TextBlob(text).sentiment.subjectivity

def polarity(text):
	return TextBlob(text).sentiment.polarity

df['subjectivity'] = df['review'].apply(subjectivity)
df['polarity'] = df['review'].apply(polarity)

def analysis(score):
	if score < 0:
		return 'Negative'
	elif score == 0:
		return 'Neutral'
	else:
		return 'Positive'

df['analysis'] = df['polarity'].apply(analysis)
df['analysis'][:20]
df['analysis'].value_counts()

#print the top 5 positive reviews and top 5 negative reviews
rank = df.sort_values(by='polarity', ascending=False)
rank['review'][:5]
rank = df.sort_values(by='polarity', ascending=True)
rank['review'][:5]

#plot the subjectivity and polarity 
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.scatter('polarity', 'subjectivity', data=df)
plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()

#show the number of positive, negative and neutral reviews
df['analysis'].value_counts().plot(kind='bar')
plt.show()
