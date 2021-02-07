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