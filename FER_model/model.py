# Make the necessary imports

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation, Dropout

from keras.regularizers import l2
from keras.optimizers import SGD




# BUILD THE MODEL

def build_model():
  ############## INPUT AND 32 BLOCK ##########################

  model = Sequential()

  model.add(Conv2D(filters=32, input_shape=(64, 64, 3),
                    kernel_size=(3,3),padding="same", activation="relu"))
  model.add(Conv2D(filters=32,kernel_size=(3,3), padding='same'))
  model.add(Activation('relu'))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))


  ################## 64 BLOCK ########################

  model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same'))
  model.add(Activation('relu'))


  model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same'))
  model.add(Activation('relu'))


  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2), padding='same'))


  ################# 128 BLOCK #######################

  model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
  model.add(Activation('relu'))

  model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
  model.add(Activation('relu'))

  model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
  model.add(Activation('relu'))

  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))



  ################# 256 BLOCK #######################

  model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
  model.add(Activation('relu'))

  model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
  model.add(Activation('relu'))

  model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
  model.add(Activation('relu'))


  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))



  ################ 256 BLOCK #######################

  model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
  model.add(Activation('relu'))

  model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
  model.add(Activation('relu'))

  model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
  model.add(Activation('relu'))


  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))


  ############## CLASSIFIER ###################

  model.add(Flatten())

  model.add(Dense(units=1024, kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
  model.add(Activation('relu'))
  model.add(Dense(units=512, kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
  model.add(Activation('relu'))
  model.add(Dropout(0.25)) 


  model.add(Dense(units=5, activation="softmax"))

# Compile the model with SGD

  opt = SGD(learning_rate=0.001, momentum=0.9, nesterov=False)
  model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


  return model
