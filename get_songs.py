import pandas as pd
import numpy as np

from scipy.spatial import distance
a = (1, 2, 3)
b = (4, 5, 6)
dst = distance.euclidean(a, b)


def most_compatible_song(song_data,targeted_feature_vector):
	min_dis = 10000.0
	min_dis_index=None

	for index,song in enumerate(song_data):
		d = distance.euclidean(song,targeted_feature_vector) 
		if d < min_dis:
			min_dis = d
			min_dis_index = index


	return min_dis_index




def read_song_data():
	df = pd.read_csv('dataset/archive/data.csv', sep=',')


	danceability = [df['danceability'].to_numpy()]
	energy = [df['energy'].to_numpy()]
	valence = [df['valence'].to_numpy()]
	tempo = [df['tempo'].to_numpy()/244]
	mode = [df['mode'].to_numpy()]

	artists = [df['artists'].to_list()]
	name = [df['name'].to_list()]


	danceability= np.array(danceability)
	song_features = danceability
	song_features = np.append(song_features,energy,axis=0)
	song_features = np.append(song_features,valence,axis=0)
	song_features = np.append(song_features,tempo,axis=0)
	song_features = np.append(song_features,mode,axis=0)

	song_features=song_features.T

	artists = np.array(artists)
	song_name_title = artists
	song_name_title = np.append(song_name_title,name,axis=0)

	song_name_title=song_name_title.T

	song_name_title = [str(x[0])+" - "+str(x[1]) for x in song_name_title]


	return song_features,song_name_title


