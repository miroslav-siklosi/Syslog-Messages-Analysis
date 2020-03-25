"""
System Log Analysis for Anomaly Detection Using Machine Learning

@author: Miroslav Siklosi
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras


"""
DATA PREPROCESSING
"""
# Importing the dataset
dataset = pd.read_csv('Data/sample_data.csv')

# Splitting the dataset into independent and dependent variables
X = dataset.iloc[:, list(range(4, 6)) + list(range(7, 84))].values
y = dataset.iloc[:, 84].values

# Taking care of missing and incorrect data
SUM = 0
MAX = 0
COUNT = 0

for i, row in enumerate(X):
    for j in [15, 16]:
        sx = str(float(X[i,j])).lower()
        if  (sx != "nan" and sx != "inf"):
            SUM = SUM + X[i,j]
            if X[i,j] > MAX:
                MAX = X[i,j]
            COUNT = COUNT + 1

AVARAGE = SUM/COUNT

for i, row in enumerate(X):
    for j in [15, 16]:
        #print(i, x)
        sx = str(float(X[i,j])).lower()
        if  sx == "nan":
            X[i, j] = AVARAGE    
        if  sx == "inf":
            X[i, j] = MAX * 10

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Creating Dummy Variables
from keras.utils import to_categorical
encoded = to_categorical(y_train)
n_labels = len(encoded[0])


""" 
MACHINE LEARNING MODULES 
"""
''' CLASSIFICATION METHODS '''
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier_LR = LogisticRegression(random_state = 0)
classifier_LR.fit(X_train, y_train)

# Predicting the Test set results
y_LR_pred = classifier_LR.predict(X_test)


# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier_KNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_KNN.fit(X_train, y_train)

# Predicting the Test set results
y_KNN_pred = classifier_KNN.predict(X_test)


#This one can take too much time to process
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier_SVM = SVC(kernel = 'linear', random_state = 0)
classifier_SVM.fit(X_train, y_train)

# Predicting the Test set results
y_SVM_pred = classifier_SVM.predict(X_test)


# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier_kSVM = SVC(kernel = 'rbf', random_state = 0)
classifier_kSVM.fit(X_train, y_train)

# Predicting the Test set results
y_kSVM_pred = classifier_kSVM.predict(X_test)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier_NB = GaussianNB()
classifier_NB.fit(X_train, y_train)

# Predicting the Test set results
y_NB_pred = classifier_NB.predict(X_test)


# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier_DTC = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_DTC.fit(X_train, y_train)

from joblib import dump, load
dump(classifier_DTC, 'DTC.joblib') 

classifier_DTC = load('DTC.joblib') 

# Predicting the Test set results
y_DTC_pred = classifier_DTC.predict(X_test)


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier_RFC = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_RFC.fit(X_train, y_train)

# Predicting the Test set results
y_RFC_pred = classifier_RFC.predict(X_test)


''' CLUSTERING METHODS '''
# K-Means Machine Learning Method
# Fitting K-Means to the dataset
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = n_labels, init = 'k-means++', random_state = 42)

# Predicting the Test set results
y_kmeans = kmeans.fit_predict(X)

# Hierarchical Clustering
# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = n_labels, affinity = 'euclidean', linkage = 'ward')

# Predicting the Test set results
y_hc = hc.fit_predict(X)


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
classifier_ANN.add(Dense(output_dim = n_labels, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier_ANN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier_ANN.fit(X_train, encoded, batch_size = 10, nb_epoch = 10)

classifier_ANN.save('ANN.h5')

# Predicting the Test set results
from keras.models import load_model
classifier_ANN = load_model('ANN.h5')
y_ANN_pred = classifier_ANN.predict(X_test)
y_ANN_pred = (y_ANN_pred > 0.5)

# Invert back to numbers
y_ANN_pred = np.argmax(y_ANN_pred, axis = 1)



''' Inverting back categorical data '''

# Invert back categories
inverted = np.argmax(encoded, axis = 1)

# Invert back labels
y_inverted = labelEncoder_y.inverse_transform(inverted)


"""
RESULTS COMPARISSION
"""
''' Confusion Matrix '''
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

# CM of Classification models
CM_LR = confusion_matrix(y_test, y_LR_pred)
CM_KNN = confusion_matrix(y_test, y_KNN_pred)
CM_SVM = confusion_matrix(y_test, y_SVM_pred)
CM_kSVM = confusion_matrix(y_test, y_kSVM_pred)
CM_NB = confusion_matrix(y_test, y_NB_pred)
CM_DTC = confusion_matrix(y_test, y_DTC_pred) 
CM_RFC = confusion_matrix(y_test, y_RFC_pred)

# CM of Clustering models
CM_KMeans = confusion_matrix(y, y_kmeans)
CM_HC = confusion_matrix(y, y_hc)

# CM of Neural Network
CM_ANN = confusion_matrix(y_test, y_ANN_pred)

# Print CMs into text file
with open("Confusion_Matrix_DTC.txt", 'w') as f:
            f.write(np.array2string(CM_DTC, separator=',', max_line_width=np.inf))
            
with open("Confusion_Matrix_ANN.txt", 'w') as f:
            f.write(np.array2string(CM_ANN, separator=',', max_line_width=np.inf))
            
with open("Confusion_Matrix_KMeans.txt", 'w') as f:
            f.write(np.array2string(CM_KMeans, separator=',', max_line_width=np.inf))






