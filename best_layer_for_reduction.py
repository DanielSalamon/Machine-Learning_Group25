'''
	Description:
		Experiment to observe the best fully connected layer of the CNN for feature reduction.
		This is done using a SVM to observe the layer that best reduces the feature vectors
'''

from readfile import *
from Classifier_Models import Classifier_Models


layers = [-2,-3,-4,-5]

accuracies = []



x = getdata()

if type(x) == bool:
	print("Dataset retrieval failed.\nCheck \"path to data.py\" to ensure path to dataset is valid")
else:
	train_images=x[0]
	train_labels=x[1]
	test_images=x[2]
	test_labels=x[3]
	val_images=x[4]
	val_labels=x[5]


for layer in layers:
	model = Classifier_Models()

	predictions,accuracy = model.svm()

	accuracies.append(accuracy)