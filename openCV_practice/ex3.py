import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt

def main():


    images = list(map(cv2.imread, glob.glob('data/multi/*.png')))
    #img = np.uint8(np.mean(np.array(images), axis=0))
    img = np.uint8(np.median(np.array(images), axis=0))
    show(img)


    #Acent
    #image = cv2.imread('data/multi/frame_00000001.png')
    #images = [cv2.imread('data/multi/frame_0000000' + str(i) + '.png') for i in range(2,10)]
    #平均値フィルタ
    '''
    img64 = np.double(image)

    for i in range(0,8):
        img64_1 = np.double(images[i])
        img64 += img64_1
    img64 /= 9.0

    img = img64.astype('uint8')
    show(img)
    '''
    #中央値フィルタ
    '''
    images_d = [np.double(cv2.imread('data/multi/frame_0000000' + str(i) + '.png')) for i in range(1,10)]
    img_d = np.double(cv2.imread('data/multi/frame_00000001.png'))
    for i,img_d_1 in enumerate(img_d):
        for j, img_d_2 in enumerate(img_d_1):
            for k, img_d_3 in enumerate(img_d_2):
                img_d[i][j][k] = np.median(np.array([images_d[n][i][j][k] for n in range(0,9)]))

    img = img_d.astype('uint8')
    show(img)
    '''


def show(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == '__main__':
    main()
