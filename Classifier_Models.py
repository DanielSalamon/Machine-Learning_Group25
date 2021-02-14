from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn import tree
from sklearn.decomposition import PCA


from sklearn.svm import SVC 
from statistics import mode

from keras.models import load_model
from keras.models import Model

class Classifier_Models():
    def __init__(self):
        self.initial_model = load_model('dataset/LittleVGG_2.h5') #CNN for feature reduction
        # self.initial_model.summary()
        
    def set_layer_for_feature_reduction(self,layer_for_feature_reduction):
        self.feature_reduction_model = Model(inputs=self.initial_model.inputs, outputs=self.initial_model.layers[layer_for_feature_reduction].output)
        
    def __preprocess(self,X_train,X_test,layer_for_feature_reduction=-2,pca=False):
        self.set_layer_for_feature_reduction(layer_for_feature_reduction)
        X_train = self.feature_reduction_model.predict(X_train)
        X_test = self.feature_reduction_model.predict(X_test)

        if pca == False:
            return X_train,X_test
        else:
            pca_model = PCA()
            if pca>min(X_train.shape[0],X_train.shape[1]):
                print(pca,"is larger than limit... setting pca value to "+str(min(X_train.shape[0],X_train.shape[1])))
                pca=min(X_train.shape[0],X_train.shape[1])
                
            pca_model = PCA(n_components=pca)
            pca_model.fit(X_train)

            X_train = pca_model.transform(X_train)
            X_test = pca_model.transform(X_test)

            # print(X_train.shape,X_test.shape)
            return X_train,X_test



    def knn(self,k,X_train=None,y_train=None, X_test=None, y_test= None ,verbose='yes',layer_for_feature_reduction=-2,pca=False):

        X_train,X_test = self.__preprocess(X_train,X_test,layer_for_feature_reduction,pca=pca)
        
        knn = KNeighborsClassifier(n_neighbors=k)
        #Train the model using the training sets
        knn.fit(X_train, y_train)
        #Predict the response for test dataset
        y_pred = knn.predict(X_test)

        if verbose == 'yes':
            acc = metrics.accuracy_score(y_test, y_pred)
            return y_pred, acc
        else:
            return y_pred


    def DT(self,X_train=None,y_train=None, X_test=None, y_test= None ,verbose='yes',split_criterion = 'gini',split_strategy = 'best',max_depth = None,layer_for_feature_reduction=-2,pca=False):
        X_train,X_test = self.__preprocess(X_train,X_test,layer_for_feature_reduction,pca=pca)

        dt = tree.DecisionTreeClassifier(criterion = split_criterion,splitter = split_strategy, max_depth=max_depth)
        #Train the model using the training sets
        dt = dt.fit(X_train, y_train)
        # tree.plot_tree(dt) 
        #Predict the response for test dataset
        y_pred = dt.predict(X_test)

        if verbose == 'yes':
            acc = metrics.accuracy_score(y_test, y_pred)
            return y_pred, acc
        else:
            return y_pred


    def svm(self,X_train=None,y_train=None,X_test=None,y_test=None,verbose='yes',layer_for_feature_reduction=-2,pca=False):
        X_train,X_test = self.__preprocess(X_train,X_test,layer_for_feature_reduction,pca=pca)
        
        if verbose == 'yes': #make classfication
            clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)

            acc = metrics.accuracy_score(y_test, y_pred)
            return y_pred, acc
        else: #make probabilistic classification
            clf = make_pipeline(StandardScaler(), SVC(gamma='auto',probability=True))
            clf.fit(X_train, y_train)

            y_pred = clf.predict_proba(X_test)
            return y_pred


    def ensemble(self,X_train,y_train,X_test,y_test,verbose='yes',layer_for_feature_reduction=-2):
        X_train,X_test = self.__preprocess(X_train,X_test,layer_for_feature_reduction)


        ###Tune these parameters
        knn_classification = knn(7,X_train=train_data_for_knn,y_train=train_output_for_knn,X_test = X_test,verbose=None,pre_processing='minmax')

        dt_classification = DT(X_train=train_data_for_DT,y_train=train_output_for_DT,X_test=X_test,split_criterion='entropy',split_strategy='best',max_depth=None,verbose=None,pre_processing=None)

        svc_classification = svm(X_train=train_data_for_DT,y_train=train_output_for_DT,X_test=X_test,verbose=None)



        predictions = []
        for x in range(0,len(X_test)):
            a = [knn_classification[x],dt_classification[x],svc_classification[x]]

            predictions.append(mode(a))

        if verbose == 'yes':
            acc = metrics.accuracy_score(y_test,predictions)
            return predictions,accuracy_score
        else:
            return predictions


# model = Classifier_Models()

