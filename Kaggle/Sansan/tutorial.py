import os
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from image_load import load
from common import showSCR
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

'''''''''''''''
Constant Values
'''''''''''''''
# %%
columns = ['company_name', 'full_name', 'position_name',
           'address', 'phone_number', 'fax',
           'mobile', 'email', 'url']
traindata_count = 500
testdata_count = 100
train_size = 0.8

'''''''''''''''
Data Loading
'''''''''''''''
#Training Data
# %%
df_train = pd.read_csv('sansan-001/train.csv')
df_train_min = df_train.sample(traindata_count, random_state=0)
X_train = load(df_train_min)
Y_train = df_train[columns].values
X_dev, X_val, Y_dev, Y_val = train_test_split(X_train, Y_train, train_size=train_size, random_state=0)
#Test data
df_test = pd.read_csv('sansan-001/test.csv')
df_test_min = df_test.sample(testdata_count, random_state=0)
X_test = load(df_test_min)

'''''''''''''''
Preprocessing
'''''''''''''''
# %%
scaler = StandardScaler()
scaler.fit(X_dev)

X_dev_scaled = scaler.transform(X_dev)
X_val_scaled = scaler.transform(X_val)


'''''''''''''''
PCA
'''''''''''''''
'''
pca = PCA(n_components=100, random_state=0)
pca.fit(X_dev_scaled)
X_dev_pca = pca.transform(X_dev_scaled)
X_val_pca = pca.transform(X_val_scaled)
print(np.sum(pca.explained_variance_ratio_))

# 累積寄与率の確認
#showSCR(X_dev_scaled)
#180ぐらいで0.95
'''
