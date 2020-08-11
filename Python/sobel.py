import cv2
import numpy as np
import os
from sklearn.externals import joblib

IMG_PATH = './jpg/yes'


def binaryFilter(image, number):
    img = image.copy()
    h, w = img.shape
    sizeh, sizew = 1, w // 15

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        x, y, ww, hh = cv2.boundingRect(contours[i])
        if area <= 50:
            img[y:y + hh][x:x + ww] = 0

    for i in range(0, h, sizeh):
        for j in range(number, w - sizew, sizew):
            sum = np.sum(image[i:i + sizeh, j:j + sizew])
            if sum // 255 >= sizeh * sizew // 5:
                img[i:i + sizeh, j:j + sizew] = 255
            else:
                img[i:i + sizeh, j:j + sizew] = 0
    return np.uint8(img)


# def goalFinder(image):
#     img = image.copy()
#     h, w = img.shape
#     sizew, sizeh = w//15, h//4
#     brightspace = np.sum(image[int(h * 0.15):int(h * 0.85) - sizeh, int(w * 0.2):int(w * 0.8) - sizew]) // 255
#     space = (int(h * 0.85) - sizeh - int(h * 0.15)) * (int(w * 0.8) - sizew - int(w * 0.2))
#     if brightspace > space // 3.5:
#         print('不是工作人员')
#         return 'no'
#     for i in range(int(h*0.15), int(h*0.85) - sizeh, sizeh // 4):
#         for j in range(int(w*0.2), int(w*0.8) - sizew, sizew // 4):
#             changeColorArr, sumR, sumC, sumRArr, sumCArr = [], [], [], [], []
#             for n in range(j + sizew - 1):
#                 changeColor = 0
#                 for m in range(i, i + sizeh - 1):
#                     if image[m][n] != image[m+1][n]:
#                         changeColor += 1
#                 changeColorArr.append(changeColor)
#                 sumCArr.append(np.sum(image[m:m + 1, n:n + sizew]))
#                 sumRArr.append(np.sum(image[m:m + sizeh, n:n + 1]))
#             sumR.append([np.sum(image[i:i + sizeh, j:j + sizew//3]) // 255,
#                             np.sum(image[i:i + sizeh, j + sizew//3:j + 2*sizew//3]) // 255,
#                             np.sum(image[i:i + sizeh, j + 2*sizew//3:j + sizew]) // 255])
#             sumC.append([np.sum(image[i:i + sizeh//3, j:j + sizew]) // 255,
#                             np.sum(image[i + sizeh//3:i + 2*sizeh//3, j:j + sizew]) // 255,
#                             np.sum(image[i + 2*sizeh//3:i + sizeh, j:j + sizew]) // 255])
#             sum = np.sum(image[i:i + sizeh, j:j + sizew]) // 255
#             nbOfColorChange = max(changeColorArr)//2
#             rectangleWork = True
#             for k in range(3):
#                 if sumR[0][k] > sum//1.5 or sumC[0][k] > sum//1.5:
#                     rectangleWork = False
#             if rectangleWork == True:
#                 if (max(sumRArr) >= sizeh*0.9 and sumRArr.count(max(sumRArr))>=sizeh/5) or (max(sumCArr) >= sizeh*0.9 and sumCArr.count(max(sumCArr))>=sizew/4):
#                     pass
#                 else:
#                     if sizeh * sizew * 0.8 >= sum >= sizeh * sizew // 10 and 7 >= nbOfColorChange >=4:
#                         cv2.rectangle(image, (j,i), (j+sizew,i+sizeh), 255, 2)
#                         print('是工作人员')
#                         return 'yes'
#     print('不是工作人员')
#     return 'no'

def goalFinder(image):
    # load
    clf = joblib.load('./pythonLogisticRegressionModel.xml')
    image = cv2.resize(image, (100, 300))
    image = np.array(image)/255
    image = image.reshape(1, 90000)
    pred = clf.predict_proba(image)
    print(pred)
    if pred[0][1] >= pred[0][0]:
        return 'yes'
    else:
        return 'no'



def logImage(IMG_PATH):
    print(IMG_PATH)
    image = cv2.imread(IMG_PATH)
    cv2.imshow('origin.jpg', image)
    image = cv2.resize(image, (200,500))
    image = cv2.medianBlur(image, 5)
    return image


def search(image):
    h, w, t = image.shape
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # # cv2.imshow('gray', gray)
    # # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # # gray = clahe.apply(gray)
    #
    # sobelx = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    # sobely = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
    #
    # # 消除垂直方向上的影响
    # for r in range(h):
    #     for c in range(w):
    #         if r>h*0.85 or r<h*0.15 or c>w*0.7 or c<w*0.3:
    #             sobely[r,c] = 0
    #         if sobelx[r, c] > 50:
    #             sobely[r, c] = 0
    # # cv2.imshow('y', sobely)
    # kernel = np.ones((3, 11), np.uint8)
    # closing = cv2.morphologyEx(sobely, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('-1', closing)
    #
    # ret, binary = cv2.threshold(closing, 120, 255, cv2.THRESH_BINARY)
    # cv2.imshow('0', binary)
    # binary = cv2.GaussianBlur(binary, (7, 7), 1.0)
    # cv2.imshow('1', binary)
    # binary = cv2.medianBlur(binary, 9)
    # cv2.imshow('2', binary)
    # ret, binary = cv2.threshold(binary, 20, 255, cv2.THRESH_BINARY)
    # cv2.imshow('3', binary)
    #
    #
    #
    # binary2 = binaryFilter(binary, 0)
    # cv2.imshow('4', binary2)

    goal = goalFinder(image)
    if goal == 'yes':
        cv2.putText(image, 'Yes', (0, 40), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 0), 2)
        # cv2.imshow('result', image)
        # cv2.waitKey(0)
        return 'yes'
    else:
        cv2.putText(image, 'No', (0, 40), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 255), 2)
        # cv2.imshow('result', image)
        # cv2.waitKey(0)
        return 'no'



if __name__ == '__main__':
    nbOfYes = 0
    nbOfNo = 0
    for f in os.listdir(IMG_PATH):
        abs_file = os.path.join(IMG_PATH,f)
        if os.path.isfile(abs_file):
            image = logImage(abs_file)
            result = search(image)
        if result == 'yes':
            nbOfYes+=1
        else:
            nbOfNo+=1
    print('共检测了', nbOfNo+nbOfYes,'张图片，共检测到',nbOfYes,'张工作人员，正确率',nbOfYes/(nbOfNo+nbOfYes))