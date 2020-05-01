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
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import datetime
from sklearn.feature_extraction.text import CountVectorizer

def extract_date(syslog):
    match = re.search(r"(\w{3} \d{1,2} \d{4} \d{2}:\d{2}:\d{2})", syslog)
    date = datetime.datetime.strptime(match.group(1), "%b %d %Y %H:%M:%S")
    date = date.timestamp()
    return date

def extract_BoW(syslogs_column):
    syslogs = []
    dates = []
    for line in syslogs_column:
        """
        1. Extract date and time, convert into timestamps (MMM DD YYYY HH:MM:SS)
        2. Extract words
        3. Merge [date and time][syslog]
        """
        dates.append(extract_date(line))    
        syslog = re.sub('[^a-zA-Z]', ' ', line) #keep letters and spaces
        syslog = syslog.lower() 
        syslog = syslog.split() #split text into words
        syslog = [PorterStemmer().stem(word) for word in syslog if not word in set(stopwords.words('english'))] #PS - keep to root of the words
        syslog = ' '.join(syslog) #merge words back into string
        syslogs.append(syslog) #
        
    cv = CountVectorizer(max_features = 40) # TODO: Amend this variable
    BagOfWords = cv.fit_transform(syslogs).toarray()
    X = np.c_[dates, BagOfWords]
    
    return X

def import_unlabelled_dataset(filename):
    
    dataset = pd.read_csv(filename, delimiter = "\t", quoting = 3, header = None, parse_dates = True, names = ["Syslog"])
    X_test = extract_BoW(dataset["Syslog"])
    
    return {"dataset": dataset, "X_test": X_test}

# Method to import labelled dataset
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

