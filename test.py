"""
System Log Analysis for Anomaly Detection Using Machine Learning

@author: Miroslav Siklosi
"""

# Importing the libraries
import argparse
import sys
import numpy as np
import ML_modules as ML
from joblib import dump, load
from sklearn.metrics import confusion_matrix
from data_preprocessing import import_dataset

# Create parser
methods_flags = (
    "LR",
    "K-NN",
    "SVM",
    "kSVM",
    "NB",
    "DTC",
    "RFC",
    "K-Means",
    "HC",
    "ANN",
)

parser = argparse.ArgumentParser(prog="PROG.py")
parser.add_argument("--mode", dest="mode", choices=["research", "prod"], required=True)
parser.add_argument("--command", dest="command", choices=["train", "test", "trainandtest"], required=True)
parser.add_argument("--method", dest="method", choices=methods_flags, required=True)
parser.add_argument("--source", dest="source", required=True)

args = parser.parse_args()

# TODO remove before publishing
print(args.command)
print(args.mode)
print(args.method)
print(args.source)

def save_classifier(classifier, method):
    if method in supervised:
        output_filename = f"classifiers/classifier_{method}.joblib"
        dump(classifier, output_filename)
    elif method in deepLearning:
        output_filename = f"classifiers/classifier_{method}.h5"
        classifier.save(output_filename)
    return output_filename


def is_dataset_source(filename):
    filename = filename.lower()
    if filename.endswith(".csv"):
        return True
    elif filename.endswith(".joblib"):
        return False
    else:
        print(f"Unknown file extension on file {filename}")
        sys.exit(1)

supervised = ("LR", "K-NN", "SVM", "kSVM", "NB", "DTC", "RFC")
unsupervised = ("K-Means", "HC")
deepLearning = ("ANN")

methods = {"LR": ML.method_LR, "K-NN": ML.method_KNN, "SVM":  ML.method_SVM, "kSVM":  ML.method_kSVM,
           "NB": ML.method_NB, "DTC":  ML.method_DTC, "RFC":  ML.method_RFC, "K-Means":  ML.method_KMeans,
           "HC": ML.method_HC, "ANN":  ML.method_ANN}

if args.mode == "research":

    if args.command == "train":
        if args.method in unsupervised:
            print("Unsupervised does not need training...exiting")
            sys.exit(1)
        
        dataset_source = is_dataset_source(args.source)
        if dataset_source:
            data = import_dataset(args.source, split=False)
            method = methods[args.method]
            classifier = method(data)
        else: # classifier
            classifier = load(args.source)
        output_filename = save_classifier(classifier, args.method)
        print(f"Trained classifier saved into file {output_filename}")
    elif args.command == "test": # test
        if not is_dataset_source(args.source):
            print(f"{args.source} is not dataset with extension .csv")
            sys.exit(1)
    
        data = import_dataset(args.source, split=False)
        if args.method in unsupervised:
            method = methods[args.method]
            y_pred = method(data)
        
            CM = confusion_matrix(data["y_test"], y_pred)
            
            # TODO print results
            
        else: # supervised, deeplearning
            classifier = load(f"classifiers/classifier_{args.method}.joblib")
            y_pred = classifier.predict(data["X_test"])
 
            if args.method in deepLearning:
                y_pred = (y_pred > 0.5)
                # Invert back to numbers
                y_pred = np.argmax(y_pred, axis = 1)
            
            CM = confusion_matrix(data["y_test"], y_pred)
        
            # TODO print results
        
    else: # trainandtest
        if not is_dataset_source(args.source):
                print(f"{args.source} is not dataset with extension .csv")
                sys.exit(1)

        if args.method in unsupervised:
            data = import_dataset(args.source, split=False)
            method = methods[args.method]
            y_pred = method(data)
        
            CM = confusion_matrix(data["y_test"], y_pred)
            
            # TODO print results
        
        else: # supervised, deeplearning
            data = import_dataset(args.source, split=True)
            method = methods[args.method]
            classifier = method(data) 
            y_pred = classifier.predict(data["X_test"])
 
            if args.method in deepLearning:
                y_pred = (y_pred > 0.5)
                # Invert back to numbers
                y_pred = np.argmax(y_pred, axis = 1)
            
            CM = confusion_matrix(data["y_test"], y_pred)
            
            # TODO print results
        """
        ''' Inverting back categorical data '''
        # Invert back categories
        invert_y = np.argmax(data["encoded_y"], axis = 1)
        invert_y_train = np.argmax(data["encoded_y_train"], axis = 1)        
        # Invert back labels
        y_inverted = data["labelEncoder_y"].inverse_transform(invert_y)
        y_train_inverted = data["labelEncoder_y"].inverse_transform(invert_y_train)
        """
else: # prod
    if args.command == "train":
        if args.method in unsupervised:
            print("Unsupervised does not need training...exiting")
            sys.exit(1)
        
        dataset_source = is_dataset_source(args.source)
        if dataset_source:
            data = import_dataset(args.source, split=False)
            method = methods[args.method]
            classifier = method(data)
        else: # classifier
            classifier = load(args.source)
        output_filename = save_classifier(classifier, args.method)
        print(f"Trained classifier saved into file {output_filename}")
    elif args.command == "test": # test
        if not is_dataset_source(args.source):
            print(f"{args.source} is not dataset with extension .csv")
            sys.exit(1)
    
        data = import_dataset(args.source, split=False)
        if args.method in unsupervised:
            method = methods[args.method]
            y_pred = method(data)
    
            # TODO print to command line
            
        else: # supervised, deeplearning
            classifier = load(f"classifiers/classifier_{args.method}.joblib")
            y_pred = classifier.predict(data["X_test"])
 
            if args.method in deepLearning:
                y_pred = (y_pred > 0.5)
                # Invert back to numbers
                y_pred = np.argmax(y_pred, axis = 1)
        
            # TODO print to command line

    else: # trainandtest
        print("trainandtest is possible only in research mode")
        sys.exit(1)


