import numpy as np
import os
import cv2
from sklearn import metrics
from sklearn import model_selection
import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential




def appendArray(images, labels, image, label):
    label = np.array([label])
    image = image.reshape(1, -1)
    if images is None:
        images = image
        labels = label
    else:
        images = np.r_[images, image]
        labels = np.r_[labels, label]

    return [images, labels]


def loadImage(IMG_PATH):
    image = cv2.imread(IMG_PATH)
    return image

folder = 'C:\\Users\\Administrator\\Desktop\\jpg\\database'
folders = [x[0] for x in os.walk(folder)]
folders = folders[1:]






A1, Y1 = None, None
A2, Y2 = [], []
for fld in folders:
    for f in os.listdir(fld):
        labels = f
        file = os.path.join(fld, f)
        if file.endswith('.jpg') or file.endswith('.JPG') or file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jepg'):
            print(file)
            image = cv2.imread(file)
            image = cv2.resize(image, (20, 60))

            image = np.array(image)
            labels = labels.split('.')
            labels = labels[0].split('_')[0]

            A1, Y1 = appendArray(A1, Y1, image/255, [labels])
            A2.append(image / 255)
            Y2.append(labels)


indexSung = np.where(Y1=='1')[0]
print('There are {0} images that are labeled 1'.format(indexSung.shape[0]))

# np.random.seed(42)
# shuffle_index = np.random.permutation(len(A1))
# # reshuffle the data and use 500 samples as training and 124 as test
# A1train, A1test = A1[shuffle_index[:len(A1)//2],:].astype(np.float32), A1[shuffle_index[len(A1)//2:],:].astype(np.float32)
# Y1train, Y1test = Y1[shuffle_index[:len(A1)//2]].astype(np.float32), Y1[shuffle_index[len(A1)//2:]].astype(np.float32)


A1, Y1 = A1.astype(np.float32), Y1.astype(np.float32)
A1_train,A1_test,Y1_train,Y1_test=model_selection.train_test_split(A1,Y1,test_size=0.1,random_state=42)   #42 表示随机状态
A2, Y2 = np.array(A2).astype(np.float32), np.array(Y2).astype(np.float32)
A2_train,A2_test,Y2_train, Y2_test=model_selection.train_test_split(A2,Y2,test_size=0.1,random_state=42)   #42 表示随机状态
Y2_train, Y2_test = keras.utils.to_categorical(Y2_train,68), keras.utils.to_categorical(Y2_test,68)
# Y1_train = np.repeat(np.arange(10),250)[:,np.newaxis]
#
# Y1train = (Y1train == '1')
# Y1test = (Y1test == '1')


# bin_clf =  LogisticRegression(max_iter=50, verbose=True)
# bin_clf.fit(A1train, Y1train)
# y_pred = bin_clf.predict(A1test)
# # save
# joblib.dump(bin_clf,'./model/lr.model')
# print('load model success!')
#
# # # load
# # clf = LogisticRegression()
# # joblib.load('./model/lr.model')
#
# print('LogisticRegression',metrics.accuracy_score(Y1test, y_pred),'\n\n')




lr=cv2.ml.LogisticRegression_create()
lr.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)            # 指定训练方法
lr.setMiniBatchSize(10)                                             # 使用每个数据的之后都更新一次模型
lr.setIterations(500)                                              # 算法结束之前的迭代次数

lr.train(A1_train, cv2.ml.ROW_SAMPLE, Y1_train)                    # 调用对象的训练方法
lr.get_learnt_thetas()                                             # 检索得到的权重　ｘ＝ｗ０ｆ０＋ｗ１ｆ１＋ｗ２ｆ２＋ｗ３ｆ３＋ｗ４ｆ４＋ｗ４　　四个特征ｆ，偏差ｗ４
#　print(lr.get_learnt_thetas())

# 测试分类器
ret1, y_pred = lr.predict (A1_train)
print (metrics.accuracy_score(Y1_train,y_pred))      # 训练数据集上的准确度

ret2, Y_pred= lr.predict(A1_test)
print (metrics.accuracy_score(Y1_test,Y_pred))

lr.save("./pythonLogisticRegressionModel2.xml")
print('load Logistic Regression model success!')




###############################  LeNet-5  ##################################


model = Sequential()
model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(60, 20, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(272, activation='relu'))
model.add(Dense(136, activation='relu'))
model.add(Dense(68, activation='softmax'))

model.summary()

model.compile(loss=keras.metrics.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

history1 = model.fit(A2_test, Y2_test, batch_size=150, epochs=15, verbose=1, validation_data=(A2_train, Y2_train))


import matplotlib.pyplot as plt

plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('model loss of LeNet-5')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save("./pythonLeNet-5Model.xml")
print('load Logistic Regression model success!')




# ##################### SVM ########################
# svm = cv2.ml.SVM_create()
# svm.setKernel(cv2.ml.SVM_LINEAR)
# svm.setType(cv2.ml.SVM_C_SVC)
# svm.setC(2.67)
# svm.setGamma(5.383)
# svm.train(A1_train, cv2.ml.ROW_SAMPLE, Y1_train)
#
# #　print(lr.get_learnt_thetas())
#
# # 测试分类器
# ret1, y_pred = svm.predict (A1_train)
# print (metrics.accuracy_score(Y1_train,y_pred))      # 训练数据集上的准确度
#
# ret2, Y_pred= svm.predict(A1_test)
# print (metrics.accuracy_score(Y1_test,Y_pred))
#
# svm.save("./pythonSVM_Model.xml")
# print('load SVM model success!')







# # save
# joblib.dump(lr,'./model/opencv.model')
# print('load model success!')







#
#
# bin_clf = DecisionTreeRegressor()
# bin_clf.fit(A1train, Y1train)
# # predictions = model.predict(A1test)
# # print(classification_report(Y1test, predictions,
# # 	target_names=le.classes_))
# y_pred = bin_clf.predict(A1test)
# print('DecisionTreeRegressor',metrics.accuracy_score(Y1test, y_pred),'\n\n')
#
# bin_clf = LinearSVC()
# bin_clf.fit(A1train, Y1train)
# # predictions = model.predict(A1test)
# # print(classification_report(Y1test, predictions,
# # 	target_names=le.classes_))
# y_pred = bin_clf.predict(A1test)
# print('LinearSVC',metrics.accuracy_score(Y1test, y_pred))




#
# from __future__ import print_function
# import tensorflow.keras as keras
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten
# from tensorflow.keras.layers import Conv2D, MaxPooling2D
# from tensorflow.keras import backend as K
#
# batch_size = 5
# num_classes = 4
# epochs = 50
#
# # input image dimensions
# img_rows, img_cols = 30, 60
#
# x_train = A1train_zoom
# x_test = A1test_zoom
# # the data, split between train and test sets
# #(x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)
#
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
#
# # convert class vectors to binary class matrices
# y_train = tensorflow.keras.utils.to_categorical(Y1train_exp, num_classes)
# y_test = tensorflow.keras.utils.to_categorical(Y1test_exp, num_classes)
#
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
#
# model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test),
#           shuffle=True)
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

