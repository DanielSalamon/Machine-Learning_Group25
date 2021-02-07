from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn import tree

from sklearn.svm import SVC 
from statistics import mode


from keras.models import load_model
from keras.models import Model

class Classifier_Models():
    def __init__(self):
        self.feature_reduction_model = load_model('my_final_model') #CNN for feature reduction

    def set_layer_for_feature_reduction(self,layer_for_feature_reduction):
        self.feature_reduction_model = Model(inputs=self.feature_reduction_model.inputs, output=model.layers[layer_for_feature_reduction].output)

    def __preprocess(self,X_train,X_test,layer_for_feature_reduction=-2):
        self.set_layer_for_feature_reduction(layer_for_feature_reduction)
        X_train = [self.feature_reduction_model.predict(i) for i in X_train]
        X_test = [self.feature_reduction_model.predict(i) for i in X_test]
        return X_train,X_test


    def knn(self,k,X_train=None,y_train=None, X_test=None, y_test= None ,verbose='yes',layer_for_feature_reduction=-2):

        X_train,X_test = self.__preprocess(X_train,X_test,layer_for_feature_reduction)
        
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


    def DT(self,X_train=None,y_train=None, X_test=None, y_test= None ,verbose='yes',split_criterion = 'gini',split_strategy = 'best',max_depth = None,layer_for_feature_reduction=-2):
        X_train,X_test = self.__preprocess(X_train,X_test,layer_for_feature_reduction)

        dt = tree.DecisionTreeClassifier(criterion = split_criterion,splitter = split_strategy, max_depth=max_depth)
        #Train the model using the training sets
        dt = dt.fit(X_train, y_train)
        tree.plot_tree(dt) 
        #Predict the response for test dataset
        y_pred = dt.predict(X_test)

        if verbose == 'yes':
            acc = metrics.accuracy_score(y_test, y_pred)
            return y_pred, acc
        else:
            return y_pred


    def svm(self,X_train=None,y_train=None,X_test=None,y_test=None,verbose='yes',layer_for_feature_reduction=-2):
        X_train,X_test = self.__preprocess(X_train,X_test,layer_for_feature_reduction)

        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if verbose == 'yes':
            acc = metrics.accuracy_score(y_test, y_pred)
            return y_pred, acc
        else:
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