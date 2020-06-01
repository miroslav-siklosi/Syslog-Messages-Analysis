"""
DATA PREPROCESSING

System Log Analysis for Anomaly Detection Using Machine Learning

MIT License

Copyright (c) 2020 Miroslav Siklosi

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text 

# Bag of Words Method
def extract_BoW(syslogs_column):
    syslogs = []
    for line in syslogs_column:    
        syslog = re.sub(r"(?:[0-9a-fA-F]:?){12}", "", line) # remove MAC Addresses
        syslog = re.sub('[^a-zA-Z]', ' ', syslog) # keep letters and spaces
        syslog = syslog.lower() 
        syslog = syslog.split() # split text into words
        syslog = [PorterStemmer().stem(word) for word in syslog if not word in set(stopwords.words('english'))] # PS - keep to root of the words
        syslog = ' '.join(syslog) # merge words back into string
        syslogs.append(syslog) 
    
    stop_words = text.ENGLISH_STOP_WORDS.union({"asa", "fw"}) # remove asa and fw from BoW
    cv = CountVectorizer(max_features = 200, stop_words = stop_words) # consider only 200 most used words
    X = cv.fit_transform(syslogs).toarray()
    return X

# Method for importing unlabelled dataset
def import_unlabelled_dataset(filename):
    # Importing the dataset
    dataset = pd.read_csv(filename, delimiter = "\t", quoting = 3, header = None, parse_dates = True, names = ["Syslog"])
    
    # Calling method BoW to create matrix X
    X_test = extract_BoW(dataset["Syslog"])
    
    return {"dataset": dataset, "X_test": X_test}

# Method for importing labelled dataset
def import_dataset(filename, split):
    # Importing the dataset
    dataset = pd.read_csv(filename, delimiter = "\t", quoting = 3, header = None, parse_dates = True, names = ["Syslog", "Anomaly"])
      
    # Splitting the dataset into independent and dependent variables 
    X = extract_BoW(dataset["Syslog"])
    y = dataset.iloc[:, -1].values
    
    # Splitting the dataset into the Training set and Test set   
    if split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    else:
        X_train = X
        X_test = X
        y_train = y
        y_test = y

    return {"dataset": dataset, 
            "X": X, "y": y, 
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test}

