#coding:utf-8
from sklearn import svm
from sklearn import datasets
from sklearn.externals import joblib

def save_model(clf, filename):
    joblib.dump(clf, filename)

def load_model(filename):
    clf = joblib.load(filename)
    return clf
