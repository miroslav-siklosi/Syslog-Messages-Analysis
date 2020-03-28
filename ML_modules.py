"""
MACHINE LEARNING MODULES 

System Log Analysis for Anomaly Detection Using Machine Learning

@author: Miroslav Siklosi
"""
# Importing the libraries


''' CLASSIFICATION METHODS '''
def method_LR(data):    
    # Fitting Logistic Regression to the Training set
    from sklearn.linear_model import LogisticRegression
    classifier_LR = LogisticRegression(random_state = 0)
    classifier_LR.fit(data["X_train"], data["y_train"])
    
    # Saving classifier (learned ML data) into file
    #dump(classifier_LR, 'LR.joblib') 
    
    return classifier_LR
    
def method_KNN(data):
    # Fitting K-NN to the Training set
    from sklearn.neighbors import KNeighborsClassifier
    classifier_KNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier_KNN.fit(data["X_train"], data["y_train"])
    
    # Saving classifier (learned ML data) into file
    #dump(classifier_KNN, 'KNN.joblib') 
    
    return classifier_KNN

def method_SVM(data):
    #This one can take too much time to process
    # Fitting SVM to the Training set
    from sklearn.svm import SVC
    classifier_SVM = SVC(kernel = 'linear', random_state = 0)
    classifier_SVM.fit(data["X_train"], data["y_train"])
    
    # Saving classifier (learned ML data) into file
    #dump(classifier_SVM, 'SVM.joblib') 
    
    return classifier_SVM

def method_kSVM(data):
    # Fitting Kernel SVM to the Training set
    from sklearn.svm import SVC
    classifier_kSVM = SVC(kernel = 'rbf', random_state = 0)
    classifier_kSVM.fit(data["X_train"], data["y_train"])
    
    # Saving classifier (learned ML data) into file
    #dump(classifier_kSVM, 'kSVM.joblib') 
    
    return classifier_kSVM

def method_NB(data):
    # Fitting Naive Bayes to the Training set
    from sklearn.naive_bayes import GaussianNB
    classifier_NB = GaussianNB()
    classifier_NB.fit(data["X_train"], data["y_train"])
    
    # Saving classifier (learned ML data) into file
    #dump(classifier_NB, 'NB.joblib') 
    
    return classifier_NB

def method_DTC(data):
    # Fitting Decision Tree Classification to the Training set
    from sklearn.tree import DecisionTreeClassifier
    classifier_DTC = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier_DTC.fit(data["X_train"], data["y_train"])
    
    # Saving classifier (learned ML data) into file
    #dump(classifier_DTC, 'DTC.joblib') 
    
    return classifier_DTC

def method_RFC(data):
    # Fitting Random Forest Classification to the Training set
    from sklearn.ensemble import RandomForestClassifier
    classifier_RFC = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier_RFC.fit(data["X_train"], data["y_train"])
    
    # Saving classifier (learned ML data) into file
    #dump(classifier_RFC, 'RFC.joblib') 
    
    return classifier_RFC


''' CLUSTERING METHODS '''

''' K-Means Machine Learning Method '''
def method_KMeans(data):
    # Fitting K-Means to the dataset
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters = data["n_labels"], init = 'k-means++', random_state = 42)
    
    # Predicting the Test set results
    y_kmeans = kmeans.fit_predict(data["X"])
    
    return y_kmeans

def method_HC(data):
    # Hierarchical Clustering
    # Fitting Hierarchical Clustering to the dataset
    from sklearn.cluster import AgglomerativeClustering
    hc = AgglomerativeClustering(n_clusters = data["n_labels"], affinity = 'euclidean', linkage = 'ward')
    
    # Predicting the Test set results
    y_hc = hc.fit_predict(data["X"])
    
    return y_hc

def method_ANN(data):
    ''' ARTIFICAL NEURAL NETWORK METHOD '''
    # Importing the Keras libraries and packages
    from keras.models import Sequential
    from keras.layers import Dense
    
    # Initialising the ANN
    classifier_ANN = Sequential()
    
    # Adding the input layer and the first hidden layer
    classifier_ANN.add(Dense(output_dim = 39, init = 'uniform', activation = 'relu', input_dim = 79))
    
    # Adding the hidden layers
    h_layers = 1
    for i in range(h_layers):
        classifier_ANN.add(Dense(output_dim = 39, init = 'uniform', activation = 'relu'))
    
    # Adding the output layer
    classifier_ANN.add(Dense(output_dim = data["n_labels"], init = 'uniform', activation = 'softmax'))
    
    # Compiling the ANN
    classifier_ANN.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    # Fitting the ANN to the Training set
    classifier_ANN.fit(data["X_train"], data["encoded"], batch_size = 10, nb_epoch = 10)
    
    # Saving classifier (learned ML data) into file
    #classifier_ANN.save('ANN.h5')
    
    return classifier_ANN

