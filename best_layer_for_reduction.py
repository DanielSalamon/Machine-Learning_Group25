'''
	Description:
		Experiment to observe the best fully connected layer of the CNN for feature reduction.
		This is done using a SVM to observe the layer that best reduces the feature vectors
'''

from readfile import *
from Classifier_Models import Classifier_Models

from matplotlib import pyplot as plt


np.set_printoptions(threshold=np.inf)
# layers = [-2,-3,-4,-5]
layers = [-4,-3,-2]#for vgg.h5
# layers = [-6,-5,-4,-3] #for best_model.h5

accuracies = []



x = getdata()

if type(x) == bool:
	print("Please copy \"Dataset\" to working directory")
else:
	train_images=x[0]
	train_labels=x[1]
	test_images=x[2]
	test_labels=x[3]
	val_images=x[4]
	val_labels=x[5]

	# val_labels=x[5][:10]
	
	for layer in layers:
		print("Using layer "+str(layer))
		model = Classifier_Models()


		predictions,accuracy = model.svm(X_train=train_images,y_train=train_labels,X_test=test_images,y_test=test_labels,layer_for_feature_reduction=layer)
		accuracies.append(accuracy)

	print(accuracies)

	plt.bar(['flat layer','fc-1','fc-2'],accuracies)

	plt.xlabel("layer")

	plt.ylabel("accuracy")

	plt.savefig("results/Network Layers accuracies using svm.png")
	plt.show()




