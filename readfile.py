import os
import numpy as np


def getdata():
	f = open("path to data.txt")

	path = f.readline()

	f.close()

	if os.path.isdir(path):
		# print("Path is valid")
		test_images = np.load(path+"/test_images.npy")
		test_labels = np.load(path+"/test_labels.npy")

		train_images = np.load(path+"/train_images.npy")
		train_labels = np.load(path+"/train_labels.npy")
		
		val_images = np.load(path+"/val_images.npy")
		val_labels = np.load(path+"/val_labels.npy")
		
		return train_images, train_labels, test_images, test_labels, val_images,val_labels
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

# 	train_images = train_images[:10]

# 	print(train_images.shape)