#coding:utf-8
import cv2
import numpy as np
import glob
import math
from matplotlib import pyplot as plt

def main():
    images = list(map(lambda src: np.double(cv2.imread(src, 0)), glob.glob('data/multi2/*.png')))
    
    for i in range(0,6):
        print(images[i][192][254])
    x = [(line[:-1].split('\t'))[1] for line in open('data/multi2/shutter.txt', encoding='utf-8')]
    x = [np.double(val) for val in x ]
    #y = [img[191][253] for img in images]
    y = [np.sum(img) for img in images]
    height = len(images[0])
    width = len(images[0][0])
    '''
    plt.plot(x, y)
    plt.show()
    print(width)
    print(height)
    '''

    img = np.double([[0 for w in range(0,width)] for h in range(0,height)])

    for h in range(0,height):
        for w in range(0,width):
            validSats = [[n,images[n][h][w]] for n in range(0,len(images)) if images[n][h][w] < 240]
            if len(validSats) == 0:
                img[h][w] = math.log(255 / np.double(x[0]))
                continue
            i = np.double(validSats[0][1]) / np.double(x[int(validSats[0][0])])
            img[h][w] = math.log(i + 60)

    maxV = img.max()
    for h in range(0,height):
        for w in range(0,width):
            img[h][w] /= (maxV / 255.0)

    plt.imshow(img,cmap='Greys_r')
    plt.show()

if __name__ == '__main__':
    main()
