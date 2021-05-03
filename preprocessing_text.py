import os
os.chdir(r"D:\Python coding\Sentiment_Classification")

import numpy as np
import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("IMDB Dataset.csv")
print(df.shape)
print(df.head())

df.sentiment.replace('positive', 1, inplace=True)
df.sentiment.replace('negative', 2, inplace=True)
print(df.head())

# Cleaning the Texts
# 1. Remove unnecesaary tags such as html tags

def clean(review_text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned, '', review_text)

df.review = df.review.apply(clean)
print(df.review[:5])

# 2. Remove Special Characters

def is_special(review_text):
    alphanum = ''
    for character in review_text:
        if character.isalnum():
            alphanum += character
        else:
            alphanum = alphanum + ' '
    return alphanum 

df.review = df.review.apply(is_special)
print(df.review[:5])

# 3. coverting to lowercase

def to_lower(review_text):
    return review_text.lower()

df.review = df.review.apply(to_lower)
print(df.review[:5])

# 4. remove stopwords

def stopwords_removal(review_text):
    stop_words = set(stopwords.words('english'))
    #print(stop_words)
    word_tokens = word_tokenize(review_text)
    #print(word_tokens)
    filtered_sentence = [w for w in word_tokens
                         if not w in stop_words]
    return filtered_sentence

df.review = df.review.apply(stopwords_removal)
print(df.review[:5])

# 5. Stemming and Lemmatization
def stemming(review_text):
    ss = SnowballStemmer('english')
    return " ".join([ss.stem(w) for w in review_text])

df.review = df.review.apply(stemming)
print(df.review[:5])

# Word to vectors using Bag of Words(BOW)
X = np.array(df.review.values)
y = np.array(df.sentiment.values)

cv = CountVectorizer(max_features = 1000)
X = cv.fit_transform(df.review).toarray()

print("Shape of X = ", X.shape)
print("Shape of y = ", y.shape)

X_df = pd.DataFrame(X)
X_df.to_csv("BOW_X.csv", index=False, header=True)
y_df = pd.DataFrame(y)
y_df.to_csv("labels.csv", index=False, header=True)
