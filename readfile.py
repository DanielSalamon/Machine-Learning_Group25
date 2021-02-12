import os
import numpy as np


def getdata():
	if os.path.isdir('dataset'):
		# print("Path is valid")
		test_images = np.load("dataset/test_images.npy")
		test_labels = np.load("dataset/test_labels.npy")

		train_images = np.load("dataset/train_images.npy")
		train_labels = np.load("dataset/train_labels.npy")
		
		val_images = np.load("dataset/val_images.npy")
		val_labels = np.load("dataset/val_labels.npy")

		# np.argmax(a, axis=1)

		#converts the labels from one-hot encoding vectors to integer classes
		return train_images, np.argmax(train_labels,axis=1),test_images,np.argmax(test_labels,axis=1),val_images,np.argmax(val_labels,axis=1)
		# return np.expand_dims(train_images,axis=0), train_labels, np.expand_dims(test_images,axis=0), test_labels, np.expand_dims(val_images,axis=0),val_labels
	else:
		# print("Path is not valid")
		return False



# x = getdata()

# if type(x) == bool:
# 	print("Dataset retrieval failed.\nCheck \"path to data.py\" to ensure path to dataset is valid")
# else:
# 	train_images=x[0]
# 	train_labels=x[1]
# 	test_images=x[2]
# 	test_labels=x[3]
# 	val_images=x[4]
# 	val_labels=x[5]


# 	print(train_images.shape)
# 	print(val_images.shape)
# 	print(test_images.shape)

# 	train_images = train_images[:10]

# 	print(train_images.shape)