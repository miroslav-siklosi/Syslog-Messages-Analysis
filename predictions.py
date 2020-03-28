"""
PREDICTIONS

System Log Analysis for Anomaly Detection Using Machine Learning

@author: Miroslav Siklosi
"""
# Importing the libraries

# Load trained classifiers
from keras.models import load_model

classifier_LR = load('Data/LR.joblib')
classifier_KNN = load('Data/KNN.joblib')
classifier_SVM = load('Data/SVM.joblib')
classifier_kSVM = load('Data/kSVM.joblib')
classifier_NB = load('Data/NB.joblib')
classifier_RFC = load('Data/RFC.joblib')
classifier_DTC = load('Data/DTC.joblib')

classifier_ANN = load_model('Data/ANN.h5')


# Predicting the Test set results
# Supervised
y_LR_pred = classifier_LR.predict(X_test)
y_KNN_pred = classifier_KNN.predict(X_test)
y_SVM_pred = classifier_SVM.predict(X_test)
y_kSVM_pred = classifier_kSVM.predict(X_test)
y_NB_pred = classifier_NB.predict(X_test)
y_RFC_pred = classifier_RFC.predict(X_test)
y_DTC_pred = classifier_DTC.predict(X_test)

# Deep Learning
y_ANN_pred = classifier_ANN.predict(X_test)
y_ANN_pred = (y_ANN_pred > 0.5)

# Invert back to numbers
y_ANN_pred = np.argmax(y_ANN_pred, axis = 1)

from keras.models import load_model
classifier_ANN = load_model('ANN.h5')


''' Inverting back categorical data '''

# Invert back categories
inverted = np.argmax(data["encoded"], axis = 1)

# Invert back labels
y_inverted = data["labelEncoder_y"].inverse_transform(inverted)
