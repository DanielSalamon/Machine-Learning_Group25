'''
	Description:
		Experiment to observe a suitable feature reduction (using PCA) on the CNN features.
'''

from readfile import *
from Classifier_Models import Classifier_Models

from matplotlib import pyplot as plt


# np.set_printoptions(threshold=np.inf)


layer = -7 #represents fc-1

pca_number_of_features = [20,50,100,False] #originally fc-1 has 512 features

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
	
	for value in pca_number_of_features:
		print("\nUsing pca value "+str(value))
		model = Classifier_Models()


		predictions,accuracy = model.svm(X_train=train_images,y_train=train_labels,X_test=val_images,y_test=val_labels,layer_for_feature_reduction=layer,pca=value)
		print("\taccuracy:",accuracy)
		accuracies.append(accuracy)

	print(accuracies)

	chart = plt.bar(['20','50','100','1024'],accuracies,color='r')

	plt.xlabel("Number of features")

	plt.ylabel("accuracy")

	plt.savefig("results/new pca experiments.png")
	plt.show()




