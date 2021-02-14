from readfile import *
from Classifier_Models import Classifier_Models
from get_songs import *
import numpy as np

from matplotlib import pyplot as plt


# np.set_printoptions(threshold=np.inf)

def get_emotion(index):
	if index==0:
		return 'neutral'
	elif index==1:
		return 'happy'
	elif index ==2:
		return 'sad'
	elif index==3:
		return 'fear'
	elif index==4:
		return 'anger'
def emotion_to_desired_music_features_mapping(probabilistic_emotions):
	map = np.array([[0.55,0.5,0.5,0.4,0.5], #neutral
					[0.9,0.8,0.95,0.5,0.95], #happy
					[0.2,0.2,0.05,0.15,0.05], #sad
					[0.2,0.3,0.75,0.2,0.6], #fear
					[0.3,0.95,0.2,0.75,0.4]]) #anger


	transform = []

	for index,emotion_vector in enumerate(probabilistic_emotions):
		transform.append(np.dot(probabilistic_emotions[index],map))

	return transform



def recommend_song(input_images):
	model = Classifier_Models()

	song_feature,song_title=read_song_data()

	recommendations = []

	predictions = model.svm(X_train=train_images,y_train=train_labels,X_test=input_images,verbose='no',layer_for_feature_reduction=-7,pca=50)


	desired_song_features = emotion_to_desired_music_features_mapping(predictions)

	for index,image in enumerate(input_images):
		x = most_compatible_song(song_feature,desired_song_features[index])

		recommendations.append(song_title[x])


	return recommendations,predictions,desired_song_features


x = getdata()

if type(x) == bool:
	print("Please copy \"Dataset\" to working directory")
else:
	train_images=x[0]#[0:50]
	train_labels=x[1]#[0:50]
	test_images=x[2][380:388]




	print(test_images.shape)
	recommendations,predictions,desired_song_features = recommend_song(test_images)

	print("recommendations\n=====================")

	print(recommendations)


	for index,image in enumerate(test_images):
		print("------------------\nImage",index)

		print("\t",recommendations[index])
		print("\t predictions:",predictions[index])
		print("\t desired_song_features:",desired_song_features[index])

		# plt.imshow(test_images[index])

		# plt.show()


	song_titles = []

	fig = plt.figure(figsize=(18,12))
	columns = 4
	rows =2

	for i in range(1,columns*rows +1):


		# txt = recommendations[i-1]


		txt = get_emotion(np.argmax(predictions[i-1])) #remove comment to display emotion classification instead of music recommendation
		if len(txt) > 30:
			txt = txt[0:30]+"..."
		fig.add_subplot(rows,columns,i).set_title(txt)
		plt.imshow(test_images[i-1])

	plt.show()



