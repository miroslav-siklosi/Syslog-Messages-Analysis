"""
DATA PREPROCESSING

System Log Analysis for Anomaly Detection Using Machine Learning

@author: Miroslav Siklosi
"""
# Importing the libraries
import pandas as pd

def import_dataset(filename):
    # Importing the dataset
    dataset = pd.read_csv(filename)
    
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
    
    return {"dataset": dataset, "X": X, "y": y, "X_train": X_train, "y_train": y_train,
            "encoded": encoded, "n_labels": n_labels, 
            "labelEncoder_y": labelEncoder_y}

