import pandas as pd
data=pd.read_csv('flask-server/twitter_training.csv',sep = ',', names=['tweet id','entity','sentiment','tweet_content'])
validation=pd.read_csv('flask-server/twitter_validation.csv',sep = ',', names=['tweet id','entity','sentiment','tweet_content'])

data = data.drop('tweet id', axis=1)
validation=validation.drop('tweet id', axis=1)
data = data.dropna(subset=['tweet_content'])

import re # Regular expressions are patterns that can be used to match and manipulate strings.

data["lowercase"]=data.tweet_content.str.lower()
data["lowercase"]=data.lowercase.apply(lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x))

import re # Regular expressions are patterns that can be used to match and manipulate strings.

validation["lowercase"]=validation.tweet_content.str.lower()
validation["lowercase"]=validation.lowercase.apply(lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x))

from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk import word_tokenize
nltk.download('stopwords')

vectorizer = CountVectorizer(
    tokenizer=word_tokenize,
    stop_words=nltk.corpus.stopwords.words('english'),
    ngram_range=(1, 1)
)

from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(data, test_size=0.2, random_state=0)

y_train = X_train['sentiment']
y_test = X_test['sentiment']

nltk.download('punkt')

X_train_vec = vectorizer.fit_transform(X_train.lowercase)
X_test_vec = vectorizer.transform(X_test.lowercase)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model1 = LogisticRegression(C=1, solver="liblinear",max_iter=200)
model1.fit(X_train_vec, y_train)

test_pred = model1.predict(X_test_vec)
print("Accuracy: ", accuracy_score(y_test, test_pred) * 100)

vectorizer2 = CountVectorizer(
    tokenizer=word_tokenize,
    stop_words=nltk.corpus.stopwords.words('english'),
    ngram_range=(1, 1)
)

def prepare_input():
  while True:
    inputstr=input("Input a string that will be classified, type 'stop' to end the program")
    inputstr=inputstr.lower()
    if inputstr=='stop':
      return
    d={'user_inp':[inputstr]}
    df=pd.DataFrame(d)
    df["user_inp"]=df.user_inp.apply(lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x))
    X_train_vec = vectorizer2.fit_transform(X_train.lowercase)
    vec=vectorizer2.transform(df.user_inp)
    pred=model1.predict(vec)
    print("Predicted sentiment:", pred[0])
prepare_input()

