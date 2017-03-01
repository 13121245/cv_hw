from __future__ import print_function
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ZeroPadding2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

batch_size = 32
nb_classes = 10
nb_epoch = 30
data_augmentation = False

# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 3

# The data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Create Keras model
model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='valid', activation='relu',
                        input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
model.add(ZeroPadding2D((1, 1)))

model.add(Convolution2D(128, 3, 3, border_mode='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ZeroPadding2D((1, 1)))

# Try to remove some convolutional layer
model.add(Convolution2D(256, 3, 3, border_mode='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ZeroPadding2D((1, 1)))

model.add(Convolution2D(512, 3, 3, border_mode='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ZeroPadding2D((1, 1)))

model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(0.2))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print('Fitting model')
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          validation_data=(X_test, Y_test))

print('Evaluating model')
score = model.evaluate(X_test, Y_test, verbose=0)

print('Score: %1.3f' % score[0])
print('Accuracy: %1.3f' % score[1])
