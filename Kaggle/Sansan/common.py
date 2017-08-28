#coding:utf-8
import os
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def showSCR(data):
    pca = PCA(n_components=200, random_state=0)
    pca.fit(data)
    ev_ratio = pca.explained_variance_ratio_
    ev_ratio = np.hstack([0,ev_ratio.cumsum()])
    plt.plot(ev_ratio)
    plt.show()
