# Machine Learning Project (Group 25)

## Abstract


## How to run the experiments
- Download the dataset folder from [link](https://github.com) to the working directory (i.e contaning this README file)
- Make sure you have Python 3.8.5 installed
- `python -m pip install --upgrade pip`
- `pip install -r requirements.txt`


## Files Description
- **Best Classifier Comparison**:<br/>
	Compares KNN, SVM and Decision Tree classifiers on a validation on the emotion recognition task<br/>
source file: `best classifier model.py`

- **Best Feature Reduction Layer**:<br/>
Experiment to observe the best fully connected layer of the CNN for feature reduction. This is done using a SVM to observe the layer that best reduces the feature vectors<br/>
source file: `best_layer_for_reduction.py`

- **Further Feature Reduction:**:<br/>
Experiment to observe a suitable feature reduction (using PCA) on the CNN features.<br/>
source file: `best_PCA.py`

- **Music Recommendation**:<br/>
Recommends songs based on the probabilistic classification of the mood of a given input image<br/>
source file: `music_recommendation.py`

## Results
**Emotion Classification Performance**\
<img src="results/classifier experiments.png" width="700">
**Emotion Recognition**\
<img src="results/emotion_recognition.png" width="700" >\
**Music Recommendation**\
<img src="results/music recommendation.png" width="700" >\
<img src="results/music recommendation1.png" width="700" >\
<img src="results/music recommendation2.png" width="700" >\


## Authors
- [Brown Ogum](https://github.com/brown532)
- [Daniel Salamon](https://github.com/DanielSalamon)
- [Jelle Bosch](https://github.com/JRABosch)

