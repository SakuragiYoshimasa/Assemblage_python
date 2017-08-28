#coding:utf-8
import cv2
import numpy as np
import glob
import math
from matplotlib import pyplot as plt

def main():
    imgL = cv2.imread('data/left.jpg',0)
    imgR = cv2.imread('data/right.jpg',0)

    print(imgL.shape)
    print(imgR.shape)

    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=5)
    disparity = stereo.compute(imgL,imgR)
    plt.imshow(disparity,'gray')
    plt.show()

if __name__ == '__main__':
    main()
    print(((int.from_bytes('生'.encode('utf-8'),'little') & int.from_bytes('死'.encode('utf-8'),'little')).to_bytes(4, 'little')).decode('utf-8'))
