#coding:utf-8
import os
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from PIL import Image

DIR_IMAGES = 'sansan-001/images'
IMG_SIZE = 100

def load(df):
    X = []
    for i, row in df.iterrows():
        img = Image.open(os.path.join(DIR_IMAGES, row.filename))
        img = img.crop((row.left, row.top, row.right, row.bottom))
        img = img.convert('L')
        img = img.resize((IMG_SIZE, IMG_SIZE), resample=Image.BICUBIC)

        x = np.asarray(img, dtype=np.float32)
        x = x.flatten()
        X.append(x)

    X = np.array(X)
    return X
