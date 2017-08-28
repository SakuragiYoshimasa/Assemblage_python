#coding:utf-8
import cv2
import numpy as np
import glob
import math
from matplotlib import pyplot as plt

def findchess(gray):
    # 格子点の3D座標を定義
    #objp = np.zeros((12*9,3), np.float32)
    objp = np.zeros((8*5,3), np.float32)
    #objp[:,:2] = np.mgrid[0:12,0:9].T.reshape(-1,2)
    objp[:,:2] = np.mgrid[0:8,0:5].T.reshape(-1,2)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # findChessboardCorners()で初期検出
    ret, corners = cv2.findChessboardCorners(gray, (5,8), None)
    #ret, corners = cv2.findChessboardCorners(gray, (9,12), None)
    # cornerSubPix()で最適化
    #if ret == True:
        #corners = cv2.cornerSubPix(gray, corners,(11,11),(-1,-1), criteria)
        #corners = cv2.cornerSubPix(gray, corners,(11,11),(-1,-1), criteria)s

    return ret, corners, objp

def synthesize(camFrame, videoFrame, corners):

    #height, width = camFrame.shape[:2]
    ##16:9
    height, width = videoFrame.shape[:2]

    pts1 = np.float32([[0,0],[width , 0],[0,height]])
    #pts2 = np.float32([corners[0],corners[3],(corners[9 * 5 ] + corners[9 * 6 ]) / 2.0])
    pts2 = np.float32([corners[0],corners[3],(corners[5 * 5 ] + corners[5 * 6 ]) / 2.0])

    #print(camFrame.shape[:2])

    # アフィン変換行列を求める
    M = cv2.getAffineTransform(pts1,pts2)

    videoFrame = videoFrame[0:height, 0:width]

    height, width = camFrame.shape[:2]
    img_afn = cv2.warpAffine(videoFrame, M, tuple(np.array([width, height])), flags=cv2.INTER_LINEAR)

    ## TODO 合成
    mask = cv2.warpAffine(np.ones(videoFrame.shape, dtype=np.uint8) , M,tuple(np.array([width, height])))


    '''
    img2_gray = cv2.cvtColor(img_afn, cv2.COLOR_BGR2GRAY)

    img_maskg = cv2.threshold(img2_gray, 220, 255, cv2.THRESH_BINARY_INV)[1]

    img_mask = cv2.merge((img_maskg,img_maskg, img_maskg))
    img_src2m = cv2.bitwise_and(img_afn, img_mask)

    img_maskn = cv2.bitwise_not(img_mask)
    img_src1m = cv2.bitwise_or(camFrame, img_maskn)

    img_dst = cv2.bitwise_or(img_src1m, img_src2m)
    '''



    #img_dst = cv2.circle(img_dst,tuple(corners[0]), 3, (0,255,0), -1)
    #img_dst = cv2.circle(img_dst,tuple(corners[8]), 3, (0,255,0), -1)
    #img_dst = cv2.circle(img_dst,tuple(corners[9 * 12 - 9]), 3, (0,255,0), -1)

    img_syn =  (1.0v - mask) * camFrame + mask * img_afn
    #img_syn = cv2.addWeighted(img_afn, 1.0, camFrame, 0.3, 2.2)
    return img_syn



def main():
    #------------------------------------------------
    #Setup
    #------------------------------------------------
    cam = cv2.VideoCapture(0)
    if cam.isOpened() is False:
        raise("IO Error")

    cap = cv2.VideoCapture('data/movie.mp4')


    while(cam.isOpened() and cap.isOpened()):
        ret, cframe = cam.read()
        #qcframe =  cv2.imread('data/chess/L/intrinsicL-00.png')

        if not ret:
            # Release the Video Device if ret is false
            cam.release()
            # Message to be displayed after releasing the device
            print("Cannot capture a frame")
            break


        ret, vframe = cap.read()
        gray = cv2.cvtColor(cframe, cv2.COLOR_BGR2GRAY)
        ret, imgPt, objPt = findchess(gray)

        #cframe =  cv2.cvtColor(cframe, cv2.COLOR_BGR2GRAY)


        if ret == True:
            #cv2.imshow('frame', cv2.drawChessboardCorners(cframe, (9,12), imgPt, ret))
            vframe =  cv2.cvtColor(vframe, cv2.COLOR_BGR2GRAY)
            cv2.imshow('frame', synthesize(gray, vframe, imgPt))
        else:
            cv2.imshow('frame', gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()
