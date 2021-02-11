import keras
from keras.datasets import fashion_mnist 
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, LeakyReLU, BatchNormalization,Dropout
from keras.models import Sequential
from keras.utils import to_categorical

from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import matplotlib.pyplot as plt

from readfile import *







#create model
model = Sequential()

model.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = 'relu', input_shape = (128, 128, 3)))
model.add(keras.layers.BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = 'relu'))
model.add(keras.layers.BatchNormalization())
model.add(Dropout(0.4))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = 'relu'))
model.add(keras.layers.BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = 'relu'))
model.add(keras.layers.BatchNormalization())

model.add(Dropout(0.4))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = 'relu'))
model.add(keras.layers.BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = 'relu'))
model.add(keras.layers.BatchNormalization())
model.add(Dropout(0.4))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(8, activation = "softmax"))

model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])





x = getdata()

train_images=x[0]
train_labels=x[1]
test_images=x[2]
test_labels=x[3]
val_images=x[4]
val_labels=x[5]


print(train_images.shape)
print(val_images.shape)


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)


#train model on training set
history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), batch_size = 16, epochs=200, verbose=2, callbacks=[es, mc])
# model.fit(train_images, train_labels, batch_size=, epochs=1000,verbose=2)


#test trained model
# test_loss, test_acc = model.evaluate(test_images, test_labels)



plt.plot(history.history['loss'], label='training')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.savefig("model_training_loss.png")
# plt.show()
plt.figure()
plt.plot(history.history['accuracy'], label='training')
plt.plot(history.history['val_accuracy'], label='validation')
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.savefig("model_training_accuracy.png")
# plt.show()

