#coding:utf-8
import cv2
import numpy as np
import glob
import math
from matplotlib import pyplot as plt



def main():
    objpoints = [] # 3d point in real world space
    imgpoints = []

    

    images = list(map(lambda src: cv2.imread(src), glob.glob('data/chess/L/*.png')))
    #print(len(images))
    objp = np.zeros((9*12,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:12].T.reshape(-1,2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    for img in images:
        #img = images[i]
        #img = cv2.imread('data/chess/L/intrinsicL-02.png')
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(image, (9,12),None)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(image,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img , (9,12), corners2,ret)
            #cv2.imshow('img',img)
            #cv2.waitKey(0)


    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image.shape[::-1],None,None)

    #内部パラメータ
    A = mtx

    #ロドリゲスから回転行列Rに直す
    R , jac = cv2.Rodrigues(rvecs[0])

    #推進ベクトル
    t = tvecs[0]

    # λx = A (RX + t)
    # c = A(RX + t) は世界座標系(チェスボードの設定に依存)→カメラ座標系で(0,0,0)への座標変換(定義より)
    # 逆に解いて X = -R^(-1)t が世界座標系でのカメラの座標


    h, w = images[0].shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    A = newcameramtx

    # undistort using func
    #images[0] = cv2.undistort(images[0], mtx, dist, None, newcameramtx)
    # undistort using remapping
    mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,A,(w,h),5)
    images[0] = cv2.remap(images[0],mapx,mapy,cv2.INTER_LINEAR)

    x,y,w,h = roi
    images[0] = images[0][y:y+h, x:x+w]

    O = [0,0,0]
    Px = [10,0,0]
    Py = [0,10,0]
    Pz = [0,0,10]
    o = A.dot(R.dot(O) + t.T[0])
    x = A.dot(R.dot(Px) + t.T[0])
    y = A.dot(R.dot(Py) + t.T[0])
    z = A.dot(R.dot(Pz) + t.T[0])

    projectiono = (int(o[0]/o[2]), int(o[1]/o[2]))
    projectionx = (int(x[0]/x[2]), int(x[1]/x[2]))
    projectiony = (int(y[0]/y[2]), int(y[1]/y[2]))
    projectionz = (int(z[0]/z[2]), int(z[1]/z[2]))

    images[0] = cv2.line(images[0],projectiono,projectionx,(255,0,0),thickness=1)
    images[0] = cv2.line(images[0],projectiono,projectiony,(255,0,0),thickness=1)
    images[0] = cv2.line(images[0],projectiono,projectiony,(255,0,0),thickness=1)


    plt.imshow(images[0])
    plt.show()

if __name__ == '__main__':
    main()
