'''
	Description:
		Experiment to observe a suitable feature reduction (using PCA) on the CNN features.
'''

from readfile import *
from Classifier_Models import Classifier_Models

from matplotlib import pyplot as plt


np.set_printoptions(threshold=np.inf)


layer = -7 #represents fc-1


pca_number_of_features = 50



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
	
	# for value in pca_number_of_features:
	# 	print("\nUsing pca value "+str(value))
	# 	model = Classifier_Models()


	# 	predictions,accuracy = model.svm(X_train=train_images,y_train=train_labels,X_test=test_images,y_test=test_labels,layer_for_feature_reduction=layer,pca=value)
	# 	print("\taccuracy:",accuracy)
	# 	accuracies.append(accuracy)

	model=Classifier_Models()

	print("\nSVM model")
	_,accuracy = model.svm(X_train=train_images,y_train=train_labels,X_test=val_images,y_test=val_labels,layer_for_feature_reduction=layer,pca=pca_number_of_features)
	print("\t accuracy:",accuracy)
	accuracies.append(accuracy)


	print("\nDT model")
	_,accuracy = model.DT(X_train=train_images,y_train=train_labels,X_test=val_images,y_test=val_labels,layer_for_feature_reduction=layer,pca=pca_number_of_features)
	print("\t accuracy:",accuracy)
	accuracies.append(accuracy)


	print("\nKNN (k=1) model")
	_,accuracy = model.knn(k=1,X_train=train_images,y_train=train_labels,X_test=val_images,y_test=val_labels,layer_for_feature_reduction=layer,pca=pca_number_of_features)
	print("\t accuracy:",accuracy)
	accuracies.append(accuracy)


	print("\nKNN (k=3) model")
	_,accuracy = model.knn(k=3,X_train=train_images,y_train=train_labels,X_test=val_images,y_test=val_labels,layer_for_feature_reduction=layer,pca=pca_number_of_features)
	print("\t accuracy:",accuracy)
	accuracies.append(accuracy)


	print("\nKNN (k=5) model")
	_,accuracy = model.knn(k=5,X_train=train_images,y_train=train_labels,X_test=val_images,y_test=val_labels,layer_for_feature_reduction=layer,pca=pca_number_of_features)
	print("\t accuracy:",accuracy)
	accuracies.append(accuracy)

	print("\nKNN (k=7) model")
	_,accuracy = model.knn(k=7,X_train=train_images,y_train=train_labels,X_test=val_images,y_test=val_labels,layer_for_feature_reduction=layer,pca=pca_number_of_features)
	print("\t accuracy:",accuracy)
	accuracies.append(accuracy)

	

	


	print(accuracies)

	chart = plt.bar(['SVM','DT','KNN (k=1)','KNN (k=3)','KNN (k=5)','KNN (k=7)'],accuracies,color='r')

	plt.xlabel("Classifier Model")

	plt.ylabel("accuracy")

	plt.savefig("results/classifier experiments.png")
	plt.show()




