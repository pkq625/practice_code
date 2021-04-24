# %tensorflow_version 1.x
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

from tensorflow.keras.datasets.mnist import load_data

data = load_data()

import matplotlib.pyplot as plt

len(data[0][1])

data[0][1] # 这个是label 一共60000个
data[0][0] # 这个是图片 一共60000个

len(data[1][0][1])
# data[1][0][1]

plt.imshow(data[1][0][1]) 
plt.show()

"""只取60000个数据，分为test和train（不用valid了qwq）"""

temp_data = data[0]

len(temp_data)

len(temp_data[0])

x_data, y_data = temp_data

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, shuffle=True, train_size=0.9)



from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dropout, Flatten, Dense, InputLayer
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.losses import CategoricalCrossentropy



def createModel():
    model = tf.keras.Sequential()
    model.add(Conv2D(
        input_shape=( 28, 28, 1),
        filters=32,
        strides=1,
        kernel_size=3,
        activation='relu',
        kernel_initializer='VarianceScaling'
    ))

    model.add(MaxPool2D(
        pool_size=2,
        strides=2
    ))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Conv2D(
        filters=64,
        strides=1,
        kernel_size=3,
        activation='relu',
        kernel_initializer='VarianceScaling'
    ))

    model.add(MaxPool2D(
        pool_size=2,
        strides=2
    ))

    model.add(BatchNormalization())
    model.add(Flatten())

    model.add(Dense(
        units=32,
        kernel_initializer='VarianceScaling',
        activation='relu'
    ))
    model.add(Dense(
        units=10,
        kernel_initializer='VarianceScaling',
        activation='softmax'
    ))
    model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['acc'])
    return model

BatchSize = 512
epochs = 10
model = createModel()

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_train.shape

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_test.shape

Y_train = tf.keras.utils.to_categorical(Y_train, 10)

Y_test = tf.keras.utils.to_categorical(Y_test, 10)

model.summary()

model.fit(X_train, Y_train, batch_size=BatchSize, epochs=epochs)
model.save('123.h5')
model.evaluate(X_test, Y_test)

