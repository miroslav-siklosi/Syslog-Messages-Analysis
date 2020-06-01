"""
System Log Analysis for Anomaly Detection Using Machine Learning

MIT License

Copyright (c) 2020 Miroslav Siklosi

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
import argparse
import sys
import numpy as np
import ML_modules as ML
from joblib import dump, load
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_preprocessing import import_dataset, import_unlabelled_dataset
from keras.models import load_model

# List of ML models flags for parser
models_flags = (
    "LR",
    "K-NN",
    "ocSVM",
    "kSVM",
    "NB",
    "DTC",
    "RFC",
    "iF",
    "LOF",
    "K-Means",
    "HC",
    "ANN"    
)

# Create parser
parser = argparse.ArgumentParser(prog="PROG.py")
parser.add_argument("--mode", dest="mode", choices=["research", "prod"], required=True)
parser.add_argument("--command", dest="command", choices=["train", "predict", "trainandpredict"], required=True)
parser.add_argument("--model", dest="model", choices=models_flags, required=True)
parser.add_argument("--source", dest="source", required=True)

args = parser.parse_args()

# Definition of ML models - used in parser due to different needs of each models
supervised = ("LR", "K-NN", "kSVM", "NB", "DTC", "RFC")
unsupervised = ("ocSVM", "iF", "LOF", "K-Means", "HC")
deepLearning = ("ANN")

# Assigning ML models to corresponding parser flags
models = {"LR": ML.model_LR, "K-NN": ML.model_KNN, "kSVM":  ML.model_kSVM, 
           "NB": ML.model_NB, "DTC":  ML.model_DTC, "RFC":  ML.model_RFC, 
           "ocSVM":  ML.model_ocSVM, "iF": ML.model_iF, "LOF": ML.model_LOF, 
           "K-Means":  ML.model_KMeans, "HC": ML.model_HC, "ANN":  ML.model_ANN}

# Method to print metrics in command line
def print_metrics(model, data, y_pred):
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(data["y_test"], y_pred)
    print(f"Accuracy of Machine Learning model {model} is", accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(data["y_test"], y_pred)
    print(f"Precision of Machine Learning model {model} is", precision)
    # recall: tp / (tp + fn)
    recall = recall_score(data["y_test"], y_pred)
    print(f"Recall of Machine Learning model {model} is", recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(data["y_test"], y_pred)
    print(f"F1-Score of Machine Learning model {model} is", f1)

# Method to print Prediction results into the text file
def print_prediction_result(data, y_pred):
    # [X_test, y_pred] Prediction is correct/Prediction is NOT correct
    X_test = data["dataset"]["Syslog"]
    y_test = data["y_test"]
    
    np.set_printoptions(threshold=np.inf)
    with open(f"Results/prediction_result.txt", 'w') as f:
        for i in range(len(X_test)):
            f.write(f"{X_test[i]} {y_pred[i]}\n")
            if y_test[i] == y_pred[i]:
                f.write("Prediction is correct\n")
            else:
                 f.write("Prediction is NOT correct\n")
    print(f"Prediction results saved into prediction_result.txt")

# Method for saving ML weights (classifier)              
def save_classifier(classifier, model):
    if model in supervised:
        output_filename = f"classifiers/classifier_{model}.joblib"
        dump(classifier, output_filename)
    elif model in deepLearning:
        output_filename = f"classifiers/classifier_{model}.h5"
        classifier.save(output_filename)
    return output_filename

# Verify if dataset to import is in correct format
def is_dataset_source(filename):
    filename = filename.lower()
    if filename.endswith(".csv"):
        return True
    elif filename.endswith(".joblib") or filename.endswith(".h5"):
        return False
    else:
        print(f"Invalid file extension on file {filename}")
        sys.exit(1)

# Method to load saved ML weight file (classifier)        
def load_classifier(filename):
    filepath = filename.lower()
    try:
        if filepath.endswith(".joblib"):
            if args.model not in supervised:
                print(f"Invalid classifier type for the {args.model} learning model")
                sys.exit(1)
            
            return load(filename)
        elif filepath.endswith(".h5"):
            if args.model not in deepLearning:
                print(f"Invalid classifier type for the {args.model} learning model")
                sys.exit(1)
            
            return load_model(filename)
        else:
            print("Classifier with unknown extension")
            sys.exit(1)
    except FileNotFoundError:
        print(f"{filepath} was not found!")
        sys.exit(1)

# PARSER
if args.mode == "research": # RESEARCH MODE
    if args.command == "train": # TRAIN
        if args.model in unsupervised:
            print("Unsupervised does not need training...exiting")
            sys.exit(1)
        
        dataset_source = is_dataset_source(args.source)
        if dataset_source:
            data = import_dataset(args.source, split=False)
            model = models[args.model]
            classifier = model(data)
        else: # classifier
            classifier = load_classifier(args.source)
        output_filename = save_classifier(classifier, args.model)
        print(f"Trained classifier saved into file {output_filename}")
    
    elif args.command == "predict": # PREDICT
        if not is_dataset_source(args.source):
            print(f"{args.source} is not dataset with extension .csv")
            sys.exit(1)
    
        data = import_dataset(args.source, split=False)
        if args.model in unsupervised:
            model = models[args.model]
            y_pred = model(data)
            
            if args.model == "ocSVM" or "iF":
                for i, row in enumerate(y_pred):                    
                    if y_pred[i] == 1:
                        y_pred[i] = 0
                    else:
                        y_pred[i] = 1       
            
        else: # Supervised, Deep Learning
            if args.model in deepLearning: # Deep Learning
                classifier = load_classifier(f"classifiers/classifier_{args.model}.h5")
                y_pred = classifier.predict(data["X_test"])
                y_pred = (y_pred > 0.5)
                # Invert back to numbers
                y_pred = np.argmax(y_pred, axis = 1)
            else:
                classifier = load_classifier(f"classifiers/classifier_{args.model}.joblib")
                y_pred = classifier.predict(data["X_test"])
                
        # Print results
        print(f"Confusion Matrix of Machine Learning model {args.model}:")
        print(confusion_matrix(data["y_test"], y_pred))
        print_metrics(args.model, data, y_pred)
        print_prediction_result(data, y_pred)
            
    else: # TRAIN AND PREDICT
        if not is_dataset_source(args.source):
                print(f"{args.source} is not dataset with extension .csv")
                sys.exit(1)

        if args.model in unsupervised:
            data = import_dataset(args.source, split=False)
            model = models[args.model]
            y_pred = model(data)
            
            if args.model == "ocSVM" or "iF":
                for i, row in enumerate(y_pred):                    
                    if y_pred[i] == 1:
                        y_pred[i] = 0
                    else:
                        y_pred[i] = 1
        
        else: # Supervised, Deep Learning
            data = import_dataset(args.source, split=True)
            model = models[args.model]
            classifier = model(data) 
            y_pred = classifier.predict(data["X_test"])
 
            if args.model in deepLearning: # Deep Learning
                y_pred = (y_pred > 0.5)
                # Invert back to numbers
                y_pred = np.argmax(y_pred, axis = 1)
            
        # Print results
        print(f"Confusion Matrix of Machine Learning model {args.model}:")
        print(confusion_matrix(data["y_test"], y_pred))
        print_metrics(args.model, data, y_pred)

else: # PRODUCTION MODE
    if args.command == "train": # TRAIN
        if args.model in unsupervised:
            print("Unsupervised does not need training...exiting")
            sys.exit(1)
        
        dataset_source = is_dataset_source(args.source)
        if dataset_source:
            data = import_dataset(args.source, split=False)
            model = models[args.model]
            classifier = model(data)
        else: # classifier
            classifier = load_classifier(args.source)
        output_filename = save_classifier(classifier, args.model)
        print(f"Trained classifier saved into file {output_filename}")
    
    elif args.command == "predict": # PREDICT
        if not is_dataset_source(args.source):
            print(f"{args.source} is not dataset with extension .csv")
            sys.exit(1)
    
        data = import_unlabelled_dataset(args.source)
        if args.model in unsupervised:
            model = models[args.model]
            y_pred = model(data)
            
            if args.model == "ocSVM" or "iF":
                for i, row in enumerate(y_pred):                    
                    if y_pred[i] == 1:
                        y_pred[i] = 0
                    else:
                        y_pred[i] = 1
                
        else: # Supervised, Deep Learning
            if args.model in deepLearning: # Deep Learning
                classifier = load_classifier(f"classifiers/classifier_{args.model}.h5")
                y_pred = classifier.predict(data["X_test"])
                y_pred = (y_pred > 0.5)
                # Invert back to numbers
                y_pred = np.argmax(y_pred, axis = 1)
            else:
                classifier = load_classifier(f"classifiers/classifier_{args.model}.joblib")
                y_pred = classifier.predict(data["X_test"])
            
        labelled_dataset = np.c_[data["dataset"], ["Anomaly" if val else "Not anomaly" for val in y_pred]]
        np.set_printoptions(threshold=np.inf)
        with open(f"Results/{args.model}_labelled.csv", 'w') as f:
            for row in labelled_dataset:
                row = np.array(list(map(lambda s: s, row)))
                r = np.array2string(row, separator='\t ', max_line_width=np.inf, formatter={'str_kind': lambda x: x})
                f.write(f"{r[1:-1]}\n")
        print(f"Labelled dataset printed out to Results/{args.model}_labelled.csv")

    else: # TRAIN AND PREDICT
        print("Train and predict is possible only in research mode")
        sys.exit(1)