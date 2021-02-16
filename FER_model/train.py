# necessary imports

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from model import build_model
import matplotlib.pyplot as plt


# import training data

train_images = np.load('train_images.npy')
train_labels = np.load('train_labels.npy')
val_images = np.load('val_images.npy')
val_labels = np.load('val_labels.npy')
test_images = np.load('test_images.npy')
test_labels = np.load('test_labels.npy')

# perform data augmentation

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(train_images)


# build the model

model = build_model()


# perform training
checkpoint = ModelCheckpoint("Little_VGG.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')  # model will be saved after every improvement

early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

history = model.fit_generator(datagen.flow(train_images, train_labels, batch_size=2),
          steps_per_epoch=len(train_images) / 2, epochs=300, validation_data=(val_images, val_labels), callbacks=[checkpoint,early])




# visualize the training history
plt.plot(history.history["accuracy"])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model training history")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["train_accuracy", "val_accuracy", "train_loss","val_loss"])
plt.savefig("training_historyVGG.png")
plt.show()
