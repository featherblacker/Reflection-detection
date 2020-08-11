import numpy as np
import os
import cv2
from sklearn import metrics
from sklearn import model_selection
import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
from sklearn.externals import joblib
from keras.models import Model
from keras.layers import Dense,Flatten
from keras.applications import vgg16
from keras import optimizers





def loadImage(IMG_PATH):
    image = cv2.imread(IMG_PATH)
    return image

folder = './database'
folders = [x[0] for x in os.walk(folder)]
folders = folders[1:]

############################ LeNet-5 ##################################
model = Sequential()
model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(60, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(272, activation='relu'))
model.add(Dense(136, activation='relu'))
model.add(Dense(68, activation='softmax'))

model.summary()
model.compile(loss=keras.metrics.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
model.save('./LeNet-5Model.xml')


################################ VGG16 ##################################
model1 = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(60, 32, 3))
model1.summary(line_length=150)

flatten = Flatten()
new_layer = Dense(68, activation='softmax')

inp = model1.input
out = new_layer(flatten(model1.output))

model1 = Model(inp, out)
model1.summary()

sgd = optimizers.SGD(lr=0.001, decay=1e-10, momentum=0.45, nesterov=True)
# sgd = optimizers.SGD(lr=0.001, decay=1e-10, momentum=0.9, nesterov=True)
model1.compile(loss=keras.metrics.categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

model1.save('./VGG16Model.xml')


############################## training data ################################
A1, Y1 = None, None

for fld in folders:
    A2, Y2 = [], []
    nb = 0
    for f in os.listdir(fld):
        if nb < 10000:
            labels = f
            file = os.path.join(fld, f)
            # if file.endswith('.jpg') or file.endswith('.JPG') or file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jepg'):
            print(file, ': ', nb)
            image = cv2.imread(file)
            image = cv2.resize(image, (32, 60))

            image = np.array(image)
            labels = labels.split('.')
            labels = labels[0].split('_')[0]

            A2.append(image / 255)
            Y2.append(labels)
            nb += 1


    indexSung = np.where(Y1=='1')[0]
    print('There are {0} images that are labeled 1'.format(indexSung.shape[0]))


    A2, Y2 = np.array(A2).astype(np.float32), np.array(Y2).astype(np.float32)
    A2_train,A2_test,Y2_train, Y2_test=model_selection.train_test_split(A2,Y2,test_size=0.1,random_state=42)   #42 表示随机状态
    Y2_train, Y2_test = keras.utils.to_categorical(Y2_train,68), keras.utils.to_categorical(Y2_test,68)


    ###############################  LeNet-5  ##################################

    model.load_weights('./LeNet-5Model.xml')
    history1 = model.fit(A2_train, Y2_train, batch_size=1000, epochs=3, verbose=1, validation_data=(A2_test, Y2_test))

    model.save('./LeNet-5Model.xml')
    print('load LeNet-5 model success!')



    ############################# VGG16 ############################

    model1.load_weights('./VGG16Model.xml')
    history2 = model1.fit(A2_train, Y2_train, batch_size=1000, epochs=2, verbose=1, validation_data=(A2_test, Y2_test))


    model1.save('./VGG16Model.xml')
    print('load VGG16 model success!')




import matplotlib.pyplot as plt

plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('model loss of LeNet-5')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('model loss of VGG16')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()