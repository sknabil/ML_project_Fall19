import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def get_labels(df, label):
    y_train = df[label]
    return y_train

def get_test_dataset():
    df_test = pd.read_csv("test.csv")
    df_test_labels = pd.read_csv("test_labels.csv")

    #ignoring the comments that have a label of -1;
    #per kaggle: value of -1 indicates it was not used for scoring; 
    ignore_idx = df_test_labels.index[df_test_labels["toxic"] == -1].tolist()

    df_test_labels = df_test_labels[df_test_labels["toxic"] != -1]
    df_test = df_test.drop(ignore_idx)
    return df_test, df_test_labels


def main():
    labels = ["toxic", "severe_toxic", "obscene", "threat",
              "insult", "identity_hate"]
    accuracy = []
    df_train = pd.read_csv("train.csv")
    df_test, df_test_labels = get_test_dataset()

    X_train = df_train["comment_text"]
    X_test = df_test["comment_text"]

    vectorizer = TfidfVectorizer()

    train_vectors = vectorizer.fit_transform(X_train)
    test_vectors = vectorizer.transform(X_test)

    for label in labels:
        y_train = get_labels(df_train, label)

        clf = RandomForestClassifier(n_estimators=100)

        clf.fit(train_vectors,np.array(y_train))

        predict = clf.predict(test_vectors)

        acc_score = accuracy_score(get_labels(df_test_labels,label),predict)
        accuracy.append(acc_score)
        
        print("Accuracy score on {}: {}"
              .format(label, acc_score))

    plt.plot(labels, accuracy)
    plt.xlabel("Toxicity")
    plt.ylabel("Accuracy")
    plt.show()

start = time.time()
main()
end = time.time()
print("Time: {}".format(end-start))
