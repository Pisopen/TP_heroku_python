import pandas as pd
import sklearn
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words
from sklearn.multiclass import OneVsRestClassifier

#Import dataset
labelDf= pd.read_csv("labels.csv",sep=',',encoding='iso-8859-1')
labelDf= labelDf.drop("Unnamed: 0",1)

clf = make_pipeline(
    TfidfVectorizer(stop_words=get_stop_words('en')),
    OneVsRestClassifier(SVC(kernel='linear', probability=True))
)

#Training model
clf = clf.fit(X=labelDf['tweet'], y=labelDf['class'])

#export model
joblib.dump(clf,'model.pkl')