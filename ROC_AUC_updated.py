import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle

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

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i, label in enumerate(labels):
        y_train = get_labels(df_train, label)

        clf = RandomForestClassifier(n_estimators=100)

        clf.fit(train_vectors,np.array(y_train))
        predict = clf.predict(test_vectors)

        y_test = get_labels(df_test_labels,label)
        fpr[i], tpr[i], _ = roc_curve(y_test,predict)
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        acc_score = round(accuracy_score(y_test,predict),
                          2)
        accuracy.append(acc_score)
        
        print("Accuracy score on {}: {}"
              .format(label, acc_score))

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test, predict)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(labels))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(labels)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(labels)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(len(labels)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend(loc="lower right")
    plt.show()


##    plt.plot(labels, accuracy)
##    plt.xlabel("Toxicity")
##    plt.ylabel("Accuracy")
##    plt.show()

start = time.time()
main()
end = time.time()
print("Time: {}".format(end-start))
