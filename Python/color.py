import cv2
import numpy as np
from skimage import data, exposure, feature
import os

IMG_PATH = './jpg/img333'


def search(IMG_PATH):
    print(IMG_PATH)
    # image = cv2.imread('./jpg/666.jpg')
    image = cv2.imread(IMG_PATH)
    cv2.imshow('origin.jpg', image)
    h, w, t = image.shape
    image = cv2.resize(image, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    h = h * 3
    w = w * 3
    image = cv2.medianBlur(image, 3)
    # cv2.imshow('origion', image)
    # image = cv2.morphologyEx(image,cv2.MORPH_OPEN,np.ones(5))
    # image = cv2.morphologyEx(image,cv2.MORPH_CLOSE,np.ones(5))
    # cv2.imshow('O&C', image)

    # image2 = np.zeros((h,w,t))
    # for i in range(h):
    #     for j in range(w):
    #         sum = int(image[i][j][0]) + int(image[i][j][1]) + int(image[i][j][2])
    #         image2[i][j][0] = int(255 * image[i][j][0] / (sum+1) + 50)
    #         image2[i][j][1] = int(255 * image[i][j][1] / (sum+1) + 50)
    #         image2[i][j][2] = int(255 * image[i][j][2] / (sum+1) + 50)
    # image2 = np.uint8(image2)
    image2 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image2 = exposure.adjust_gamma(image2, 1.2)

    anyGreen = False
    greenImage = np.zeros((h, w, t))
    for i in range(h):
        for j in range(w):
            if 77 >= image2[i][j][0] >= 35 and 255 >= image2[i][j][1] >= 25 and 255 >= image2[i][j][2] >= 25:
                greenImage[i][j][:] = [255, 255, 255]
                anyGreen = True

    if anyGreen:
        greenImage = cv2.cvtColor(np.uint8(greenImage), cv2.COLOR_BGR2GRAY)
        # greenImage = cv2.medianBlur(greenImage, 3)
        greenBinary = cv2.adaptiveThreshold(greenImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 25)

        contours, hierarchy = cv2.findContours(greenImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contoursNew = []
        areaNew = []
        con = []  # con[i][0]:conx, con[i][1]:cony, con[i][2]:conw, con[i][3]:conh
        conNew = []
        for i, cnt in enumerate(contours):
            con.append(cv2.boundingRect(cnt))
            area = cv2.contourArea(cnt)
            if area > 100:
                contoursNew.append(cnt)
                image = cv2.rectangle(image, (con[i][0], con[i][1]), (con[i][0] + con[i][2], con[i][1] + con[i][3]),
                                      (255, 0, 0), 2)
                areaNew.append(area)
        areaNew = np.array(areaNew)
        arr = []
        for i in range(len(contoursNew)):
            for j in range(len(contoursNew[i])):
                arr.append(contoursNew[i][j][0])
        arr = np.array(arr)

        if arr.size > 0:
            x, y, recw, rech = cv2.boundingRect(np.array(arr))
            # image = cv2.rectangle(image, (x, y), (x + recw, y + rech), (0, 0, 255), 2)
            rectSize = rech * recw

            if rectSize * 0.7 > areaNew.sum() >= rectSize / 10 and 8 >= areaNew.size >= 2:
                cv2.putText(image, 'Yes', (0, 40), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 0), 2)
                print('是工作人员')
            else:
                cv2.putText(image, 'No', (0, 40), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 255), 2)
                cv2.imshow('im.jpg', image)
                cv2.waitKey()

        else:
            cv2.putText(image, 'No', (0, 40), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 255), 2)
            cv2.imshow('im.jpg', image)
            cv2.waitKey()

            # grayImage = np.zeros((h, w, t))
            # for i in range(x, (x+ recw)):
            #     for j in range(y,(y+rech)):
            #         if 180 >= image2[i][j][0] >= 0 and 255>= image2[i][j][1] >= 0 and 46>= image2[i][j][2] >= 0:
            #         # if 180>=image2[i][j][0]>=0 and 43>=image2[i][j][1]>=0 and 160>=image2[i][j][2]>=46:
            #             grayImage[i][j][:]=[255,255,255]
            # cv2.imshow("gray", grayImage)
            # grayImage = cv2.cvtColor(np.uint8(grayImage),cv2.COLOR_BGR2GRAY)
            # grayImage = cv2.medianBlur(grayImage,3)
            # grayBinary = cv2.adaptiveThreshold(grayImage,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,10)
            # contoursGray, hierarchyGray = cv2.findContours(grayBinary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.imshow("grayBinary", grayBinary)
            # cv2.drawContours(image, contoursGray, -1, (0, 0, 255), 1)
            #
            # # for i in range(hg):
            # #     for j in range(wg):
            # #         if grayBinary[i][j] == 0:
            # #             image[i][j] = [0, 0, 0]
    else:
        cv2.putText(image, 'No', (0, 40), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 255), 2)
        cv2.imshow('im.jpg', image)
        cv2.waitKey()


if __name__ == '__main__':
    for f in os.listdir(IMG_PATH):
        abs_file = os.path.join(IMG_PATH, f)
        if os.path.isfile(abs_file):
            search(abs_file)
