"""
MACHINE LEARNING MODULES 

System Log Analysis for Anomaly Detection Using Machine Learning

MIT License

Copyright (c) 2020 Miroslav Siklosi

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

''' SUPERVISED ALGORITHMS '''

''' Logistic Regression '''
def model_LR(data):    
    from sklearn.linear_model import LogisticRegression
    
    classifier_LR = LogisticRegression(penalty='none', random_state = 0)
    classifier_LR.fit(data["X_train"], data["y_train"])
    
    return classifier_LR

''' K-NN '''    
def model_KNN(data):
    from sklearn.neighbors import KNeighborsClassifier
    
    classifier_KNN = KNeighborsClassifier(n_neighbors = 5, metric = 'hamming')
    classifier_KNN.fit(data["X_train"], data["y_train"])
    
    return classifier_KNN

''' Kernel SVM '''
def model_kSVM(data):
    from sklearn.svm import SVC
    
    classifier_kSVM = SVC(kernel = 'rbf', random_state = 0)
    classifier_kSVM.fit(data["X_train"], data["y_train"])
    
    return classifier_kSVM

''' Naive Bayes '''
def model_NB(data):
    from sklearn.naive_bayes import GaussianNB
    
    classifier_NB = GaussianNB()
    classifier_NB.fit(data["X_train"], data["y_train"])
    
    return classifier_NB

''' Decision Tree Classification '''
def model_DTC(data):
    from sklearn.tree import DecisionTreeClassifier
    
    classifier_DTC = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier_DTC.fit(data["X_train"], data["y_train"])
    
    return classifier_DTC

''' Random Forest Classification '''
def model_RFC(data):
    from sklearn.ensemble import RandomForestClassifier
    
    classifier_RFC = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)
    classifier_RFC.fit(data["X_train"], data["y_train"])
    
    return classifier_RFC


''' UNSUPERVISED ALGORITHMS '''

''' One-class SVM '''
def model_ocSVM(data):
    from sklearn.svm import OneClassSVM
    
    ocSVM = OneClassSVM(kernel="rbf")
    y_pred = ocSVM.fit_predict(data["X_test"])
    
    return y_pred

''' Isolation Forest '''
def model_iF(data):
    from sklearn.ensemble import IsolationForest
    
    iF = IsolationForest(random_state=0)
    y_pred = iF.fit_predict(data["X_test"])
    
    return y_pred

''' Local Outlier Factor '''
def model_LOF(data):
    from sklearn.neighbors import LocalOutlierFactor
    
    lof = LocalOutlierFactor(metric = 'hamming')
    y_pred = lof.fit_predict(data["X_test"])
    
    return y_pred

''' K-Means Machine Learning Model '''
def model_KMeans(data):
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters = 2, init = 'k-means++', algorithm = 'full', random_state = 42)
    y_pred = kmeans.fit_predict(data["X_test"])
    
    return y_pred

''' Hierarchical Clustering '''
def model_HC(data):
    from sklearn.cluster import AgglomerativeClustering
    
    hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
    y_pred = hc.fit_predict(data["X_test"])
    
    return y_pred

''' NEURAL NETWORKS '''

''' ARTIFICIAL NEURAL NETWORK MODEL '''
def model_ANN(data):
    from keras.models import Sequential
    from keras.layers import Dense
    
    # Initialising the ANN
    classifier_ANN = Sequential()
    
    # Adding the input layer and the first hidden layer
    classifier_ANN.add(Dense(activation="relu", input_dim=200, units=101, kernel_initializer="uniform"))
    
    # Adding the hidden layers
    h_layers = 10
    for i in range(h_layers):
        classifier_ANN.add(Dense(activation="relu", units=101, kernel_initializer="uniform"))
    
    # Adding the output layer
    classifier_ANN.add(Dense(activation="sigmoid", units=2, kernel_initializer="uniform"))
    
    # Compiling the ANN
    classifier_ANN.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    
    # Fitting the ANN to the Training set
    classifier_ANN.fit(data["X_train"], data["y_train"], batch_size = 10, epochs = 10)
    
    return classifier_ANN

