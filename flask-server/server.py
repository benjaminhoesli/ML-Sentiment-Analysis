from flask import Flask, request, jsonify
from flask_cors import CORS

app=Flask(__name__)
CORS(app)



import pandas as pd
data=pd.read_csv('/Users/benjin8or/Desktop/Howard/Semester 7/Machine Learning/Sentiment project/flask-server/twitter_training.csv',sep = ',', names=['tweet id','entity','sentiment','tweet_content'])
validation=pd.read_csv('/Users/benjin8or/Desktop/Howard/Semester 7/Machine Learning/Sentiment project/flask-server/twitter_validation.csv',sep = ',', names=['tweet id','entity','sentiment','tweet_content'])

data = data.drop('tweet id', axis=1)
validation=validation.drop('tweet id', axis=1)
data = data.dropna(subset=['tweet_content'])

import re # Regular expressions are patterns that can be used to match and manipulate strings.

data["lowercase"]=data.tweet_content.str.lower()
data["lowercase"]=data.lowercase.apply(lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x))

data = data[data['sentiment'] != 'Irrelevant']
data = data.drop('entity', axis=1)

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


vectorizer2 = CountVectorizer(
    tokenizer=word_tokenize,
    stop_words=nltk.corpus.stopwords.words('english'),
    ngram_range=(1, 1)
)



from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create individual models
logistic_model = LogisticRegression(C=1, solver="liblinear", max_iter=200)
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=0)

voting_classifier_soft = VotingClassifier(estimators=[
    ('logistic', logistic_model),
    ('random_forest', random_forest_model)
], voting='soft')  


voting_classifier_soft.fit(X_train_vec, y_train)

test_pred = voting_classifier_soft.predict(X_test_vec)

accuracy = accuracy_score(y_test, test_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))



def model(user_input):
    d={'user_inp':[user_input]}
    df=pd.DataFrame(d)
    df["user_inp"]=df.user_inp.apply(lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x))
    X_train_vec = vectorizer2.fit_transform(X_train.lowercase)
    vec=vectorizer2.transform(df.user_inp)
    pred=voting_classifier_soft.predict(vec)


    return(str(pred[0]))


@app.route("/members")
def members():
    return {"members":["Member1", "Member2", "Member3"]}


@app.route("/sentiment", methods=["POST"])
def sentiment():
    if request.method == "POST":
        user_input = request.json['user_inp']
        result = model(user_input)
        print(result)
        return (result)
        
    else:
        return jsonify({"error": "Invalid request method"})

if __name__=="__main__":
    app.run(debug=True)

