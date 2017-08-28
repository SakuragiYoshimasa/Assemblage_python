import cv2
import numpy as np
from matplotlib import pyplot as plt

def main():
    img = cv2.imread('data/whitebalance.jpg')

    show(img)

    center = img[1000 : 1200, 400 : 600] #y,x
    before = cv2.sumElems(center)
    scale = np.double(max(before[0:3])) / before[0:3]
    
    img = np.uint8(img * scale)

    show(img)

def show(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    plot(img)

def plot(img):
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()

if __name__ == '__main__':
    main()
