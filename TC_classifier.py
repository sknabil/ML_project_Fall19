import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
import nltk
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import csv

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text


def Generate(name, pipeline, writer):
    Print("Executing ", name)
                
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    acc_list = []
    for i,category in enumerate(categories):
        print('Processing {}'.format(category))
        # train the model using X_dtm & y
        pipeline.fit(X_train, train[category])
        # compute the testing accuracy
        prediction = pipeline.predict(X_test)
        print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))
        acc_list.append(accuracy_score(test[category], prediction))
        fpr[i], tpr[i], _ = roc_curve(test[category],prediction)
        roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(test[category], predict)
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    writer.writerow(accuracy_list)    
        
    return fpr, tpr, roc_auc


def Plot_AUROC_curve(fpr, tpr, roc_auc, name):
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(categories))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(categories)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(categories)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure()
    lw = 2

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(len(categories)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend(loc="lower right")
    plt.savefig('auroc_'+name+'.png')


with open('Result_accuracy.csv', mode='w') as res:
    writer = csv.writer(res, delimiter='\t')
    writer.writerow(['Class', 'Accuracy'])

    df = pd.read_csv("train.csv", encoding = "ISO-8859-1")

    df_toxic = df.drop(['id', 'comment_text'], axis=1)
    counts = []
    categories = list(df_toxic.columns.values)
    for i in categories:
        counts.append((i, df_toxic[i].sum()))


    rowsums = df.iloc[:,2:].sum(axis=1)
    x=rowsums.value_counts()

    #barplot
    plt.figure(figsize=(8,5))
    ax = sns.barplot(x.index, x.values)
    plt.title("Multiple categories per comment")
    plt.ylabel('# of Occurrences', fontsize=12)
    plt.xlabel('# of categories', fontsize=12)
    plt.savefig('barplot.png')

    lens = df.comment_text.str.len()
    lens.hist(bins = np.arange(0,5000,50))

    categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df['comment_text'] = df['comment_text'].map(lambda com : clean_text(com))
    #print(df['comment_text'][0])

    train, test = train_test_split(df, random_state=42, test_size=0.33, shuffle=True)

    #print("TRAIN::", train)
    X_train = train.comment_text
    X_test = test.comment_text
    #print(X_train.shape)
    #print(X_test.shape)

    classifier_list = ['Multinomial_Naive_Bayes', 'Linear_SVC', 'Logistic_Regression', 'Random_Forest', 'Decision_Tree']

    for k,classifier in enumerate(classifier_list):
        if k==0:
            pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                    ('clf', OneVsRestClassifier(MultinomialNB(
                        fit_prior=True, class_prior=None))),
                ]) 
        elif k==1:
            pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                    ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
                ])
        elif k==2:
            pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
                ])
        elif k==3:
            pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                    ('clf', OneVsRestClassifier(RandomForestClassifier(n_estimators=100), n_jobs=1)),
                ])

        elif k==4:
            pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                    ('clf', OneVsRestClassifier(DecisionTreeClassifier(), n_jobs=1)),
                ])

        fpr, tpr, roc_auc = Generate(classifier, pipeline, writer)
        Plot_AUROC_curve(fpr, tpr, roc_auc, classifier)
