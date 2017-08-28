
import cv2
import sys
import numpy
import matplotlib
from IPython.display import clear_output
import matplotlib.pyplot as plt

def main():

    '''
    print("Python    : %s " % sys.version)
    print("OpenCV    : %s " % cv2.__version__)
    print("Numpy     : %s " % numpy.__version__)
    print("Matplotlib: %s " % matplotlib.__version__)
    '''
    #画像表示
    img = cv2.imread("data/face.jpg")
    #cv2.imshow('res', img)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #show()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Turn off the axis

    sobeled_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    sobeled_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1)

    cv2.imshow('gray_sobel_edge', sobeled_x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('gray_sobel_edge', sobeled_y)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    gray_abs_sobelx = cv2.convertScaleAbs(sobeled_x)
    gray_abs_sobely = cv2.convertScaleAbs(sobeled_y)

    gray_sobel_edge = cv2.addWeighted(gray_abs_sobelx,0.5,gray_abs_sobely,0.5,0)
    cv2.imshow('gray_sobel_edge',gray_sobel_edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    laplace = cv2.Laplacian(gray, cv2.CV_64F)

    img64 = numpy.double(img)
    abs(laplace)
    img64[:,:,2] += cv2.max(abs(laplace) - 200, 0) * 4
    numpy.clip(img64, 0, 255, out = img64)
    img = img64.astype('uint8')

    gray_sobel_edge = cv2.cvtColor(gray_sobel_edge, cv2.COLOR_GRAY2BGR)
    img = cv2.addWeighted(img, 0.5,gray_sobel_edge,0.5,0)

    plt.axis('off')
    plt.title("Input Stream")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()



    '''
    capture = cv2.VideoCapture(0)
    if capture.isOpened() is False:
        raise("IO Error")

    try:
        while(True):
            # Capture frame-by-frame
            ret, frame = capture.read()
            if not ret:
                # Release the Video Device if ret is false
                capture.release()
                # Message to be displayed after releasing the device
                print("Cannot capture a frame")
                break
            # Convert the image from OpenCV BGR format to matplotlib RGB format
            # to display the image
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # Turn off the axis

            sobeled_x = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=5)
            sobeled_y = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=5)

            laplace = cv2.Laplacian(gray, cv2.CV_64F)

            img64 = numpy.double(frame)
            abs(laplace)
            img64[:,:,2] += cv2.max(abs(laplace) - 200, 0) * 4
            numpy.clip(img64, 0, 255, out = img64)
            img = img64.astype('uint8')


            plt.axis('off')
            plt.title("Input Stream")
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()
            # Display the frame until new frame is available
            clear_output(wait=True)
    except KeyboardInterrupt:
        # Release the Video Device
        capture.release()
        # Message to be displayed after releasing the device
        print("Released Video Resource")
    '''

if __name__ == '__main__':
    main()
