"""
PREDICTIONS

System Log Analysis for Anomaly Detection Using Machine Learning

MIT License

Copyright (c) 2020 Miroslav Siklosi

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
# Importing the libraries

# Load trained classifiers
import numpy as np
from joblib import load
from keras.models import load_model
import ML_modules

# Load the classifiers
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
y_LR_pred = classifier_LR.predict(data["X_test"])
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



''' Inverting back categorical data '''

# Invert back categories
invert_y = np.argmax(data["encoded_y"], axis = 1)
invert_y_train = np.argmax(data["encoded_y_train"], axis = 1)

# Invert back labels
y_inverted = data["labelEncoder_y"].inverse_transform(invert_y)
y_train_inverted = data["labelEncoder_y"].inverse_transform(invert_y_train)
