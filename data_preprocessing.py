"""
DATA PREPROCESSING

System Log Analysis for Anomaly Detection Using Machine Learning

MIT License

Copyright (c) 2020 Miroslav Siklosi

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
# Importing the libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

def import_unlabelled_dataset(filename):
    dataset = pd.read_csv(filename)
    
    X_test = dataset.iloc[:, list(range(4, 6)) + list(range(7, 84))].values

    # TODO: Standardize this part
    SUM = 0
    MAX = 0
    COUNT = 0
    
    for i, row in enumerate(X_test):
        for j in [15, 16]:
            sx = str(float(X_test[i,j])).lower()
            if  (sx != "nan" and sx != "inf"):
                SUM = SUM + X_test[i,j]
                if X_test[i,j] > MAX:
                    MAX = X_test[i,j]
                COUNT = COUNT + 1
    
    AVARAGE = SUM/COUNT
    
    for i, row in enumerate(X_test):
        for j in [15, 16]:
            sx = str(float(X_test[i,j])).lower()
            if  sx == "nan":
                X_test[i, j] = AVARAGE    
            if  sx == "inf":
                X_test[i, j] = MAX * 10 # max hodnoty float("inf")
    
    return {"X_test": X_test}

def import_dataset(filename, split, separate_y):
    # Importing the dataset
    dataset = pd.read_csv(filename)
    
    # Splitting the dataset into independent and dependent variables
    X = dataset.iloc[:, list(range(4, 6)) + list(range(7, 84))].values
    y = dataset.iloc[:, -1].values
    
    # TODO Take care of date and time values - encode into timestamps
    
    # Taking care of missing and incorrect data
    # TODO: Standardize this part
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
            sx = str(float(X[i,j])).lower()
            if  sx == "nan":
                X[i, j] = AVARAGE    
            if  sx == "inf":
                X[i, j] = MAX * 10 # max hodnoty float("inf")
    
    # Encoding categorical data    
    labelEncoder_y = LabelEncoder()
    y = labelEncoder_y.fit_transform(y)
    
    # Splitting the dataset into the Training set and Test set   
    if split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    else:
        X_train = X
        X_test = X
        y_train = y
        y_test = y
    
    # Creating Dummy Variables
    encoded_y = to_categorical(y)
    encoded_y_train = to_categorical(y_train)
    n_labels_y = len(encoded_y[0])
    n_labels_y_train = len(encoded_y_train[0])
    
    return {"dataset": dataset, 
            "X": X, "y": y, 
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test, 
            "encoded_y": encoded_y, "n_labels_y": n_labels_y,
            "encoded_y_train": encoded_y_train, "n_labels_y_train": n_labels_y_train,
            "labelEncoder_y": labelEncoder_y}

