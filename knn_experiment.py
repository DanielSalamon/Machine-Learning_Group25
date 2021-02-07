# from readfile import *


# x = getdata()

# print("here")


# if type(x) == bool:
# 	print("failed")
# else:
# 	train_images=x[0]
# 	train_labels=x[1]
# 	test_images=x[2]
# 	test_labels=x[3]
# 	val_images=x[4]
# 	val_labels=x[5]


# 	print(train_images.shape)
# 	print(train_labels.shape)


# 	print(test_images.shape)
# 	print(test_labels.shape)
	

# 	print(val_images.shape)
# 	print(val_labels.shape)

# 	np.save("image_npy.npy",train_images[0])



# import numpy as np
# from sklearn.mixture import GaussianMixture
# X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
# gm = GaussianMixture(n_components=2, random_state=0).fit(X)
# print(gm.means_)
# print(gm.predict([[9, 0], [12, 3]]))


X = [   [[1,1,2],[1,3,3]],  [[3,2,1],[7,4,2]]]

X = [x[2] for x in X[0]]
print(X)



# y = [4, 4, 1, 1]
# from sklearn.neighbors import KNeighborsClassifier


# neigh = KNeighborsClassifier(n_neighbors=3)
# neigh.fit(X, y)
# print(neigh.predict([[1.1]]))