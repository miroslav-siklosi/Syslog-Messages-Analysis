"""
System Log Analysis for Anomaly Detection Using Machine Learning

MIT License

Copyright (c) 2020 Miroslav Siklosi

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

# Importing the libraries
import argparse
import sys
import numpy as np
import ML_modules as ML
from joblib import dump, load
from sklearn import neighbors, ensemble, decomposition, datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_preprocessing import import_dataset, import_unlabelled_dataset
from keras.models import load_model
from time import time
import matplotlib.pyplot as plt
from matplotlib import offsetbox

# Create parser
methods_flags = (
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

parser = argparse.ArgumentParser(prog="PROG.py")
parser.add_argument("--mode", dest="mode", choices=["research", "prod"], required=True)
parser.add_argument("--command", dest="command", choices=["train", "test", "trainandtest"], required=True)
parser.add_argument("--method", dest="method", choices=methods_flags, required=True)
parser.add_argument("--source", dest="source", required=True)


args = parser.parse_args(["--mode", "research", "--method", "HC", "--command", "trainandtest", "--source", "Datasets\logs_sample.csv"]) # TODO remove before publishing
#args = parser.parse_args(["--mode", "research", "--method", "LR", "--command", "train", "--source", "Datasets\logs_sample.csv"]) # TODO remove before publishing
#args = parser.parse_args(["--mode", "research", "--method", "LR", "--command", "test", "--source", "Datasets\logs_sample1.csv"]) # TODO remove before publishing
#args = parser.parse_args(["--mode", "prod", "--method", "iF", "--command", "test", "--source", "Datasets\logs_sample2.csv"]) # TODO remove before publishing
#args = parser.parse_args()

# TODO remove before publishing
print(args.mode)
print(args.command)
print(args.method)
print(args.source)


supervised = ("LR", "K-NN", "kSVM", "NB", "DTC", "RFC")
unsupervised = ("ocSVM", "iF", "LOF", "K-Means", "HC")
deepLearning = ("ANN")

methods = {"LR": ML.method_LR, "K-NN": ML.method_KNN, "kSVM":  ML.method_kSVM, 
           "NB": ML.method_NB, "DTC":  ML.method_DTC, "RFC":  ML.method_RFC, 
           "ocSVM":  ML.method_ocSVM, "iF": ML.method_iF, "LOF": ML.method_LOF, 
           "K-Means":  ML.method_KMeans, "HC": ML.method_HC, "ANN":  ML.method_ANN}

def print_metrics(method, data, y_pred):
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(data["y_test"], y_pred)
    print(f"Accuracy of Machine Learning method {method} is", accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(data["y_test"], y_pred)
    print(f"Precision of Machine Learning method {method} is", precision)
    # recall: tp / (tp + fn)
    recall = recall_score(data["y_test"], y_pred)
    print(f"Recall of Machine Learning method {method} is", recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(data["y_test"], y_pred)
    print(f"F1-Score of Machine Learning method {method} is", f1)

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
   
# def plot_embedding(data, title=None):
#     digits = datasets.load_digits(n_class=6)
    
#     print(data["X_test"])
#     x_min, x_max = np.min(data["X_test"], 0), np.max(data["X_test"], 0)
#     X = (data["X_test"] - x_min) / (x_max - x_min)

#     plt.figure()
#     ax = plt.subplot(111)
#     for i in range(X.shape[0]):
#         plt.text(X[i, 0], X[i, 1], str(data["y_pred[i]"]),
#                  color=plt.cm.Set1(data["y_pred[i]"] / 10.),
#                  fontdict={'weight': 'bold', 'size': 9})

#     if hasattr(offsetbox, 'AnnotationBbox'):
#         # only print thumbnails with matplotlib > 1.0
#         shown_images = np.array([[1., 1.]])  # just something big
#         for i in range(X.shape[0]):
#             dist = np.sum((X[i] - shown_images) ** 2, 1)
#             if np.min(dist) < 4e-3:
#                 # don't show points that are too close
#                 continue
#             shown_images = np.r_[shown_images, [X[i]]]
#             imagebox = offsetbox.AnnotationBbox(
#                 offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
#                 X[i])
#             ax.add_artist(imagebox)
#     plt.xticks([]), plt.yticks([])
#     if title is not None:
#         plt.title(title)
        
# def plot_graph(data):
#     print("Computing Totally Random Trees embedding")
#     hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,
#                                            max_depth=5)
#     t0 = time()
#     X_transformed = hasher.fit_transform(data["X_test"])
#     pca = decomposition.TruncatedSVD(n_components=2)
#     X_reduced = pca.fit_transform(X_transformed)
    
#     plot_embedding(X_reduced,
#                    "Random forest embedding of the digits (time %.2fs)" %
#                    (time() - t0))

               
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
    elif filename.endswith(".joblib") or filename.endswith(".h5"):
        return False
    else:
        print(f"Invalid file extension on file {filename}")
        sys.exit(1)
        
def load_classifier(filename):
    filepath = filename.lower()
    try:
        if filepath.endswith(".joblib"):
            if args.method not in supervised:
                print(f"Invalid classifier type for the {args.method} learning method")
                sys.exit(1)
            
            return load(filename)
        elif filepath.endswith(".h5"):
            if args.method not in deepLearning:
                print(f"Invalid classifier type for the {args.method} learning method")
                sys.exit(1)
            
            return load_model(filename)
        else:
            print("Classifier with unknown extension")
            sys.exit(1)
    except FileNotFoundError:
        print(f"{filepath} was not found!")
        sys.exit(1)

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
            classifier = load_classifier(args.source)
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
            
            if args.method == "ocSVM" or "iF":
                for i, row in enumerate(y_pred):                    
                    if y_pred[i] == 1:
                        y_pred[i] = 0
                    else:
                        y_pred[i] = 1
        
            # Print results
            print(f"Confusion Matrix of Machine Learning Method {args.method}:")
            print(confusion_matrix(data["y_test"], y_pred))
            print_metrics(args.method, data, y_pred)
            print_prediction_result(data, y_pred)            
            
        else: # supervised, deeplearning
            if args.method in deepLearning:
                classifier = load_classifier(f"classifiers/classifier_{args.method}.h5")
                y_pred = classifier.predict(data["X_test"])
                y_pred = (y_pred > 0.5)
                # Invert back to numbers
                y_pred = np.argmax(y_pred, axis = 1)
            else:
                classifier = load_classifier(f"classifiers/classifier_{args.method}.joblib")
                y_pred = classifier.predict(data["X_test"])
                
            # Print results
            print(f"Confusion Matrix of Machine Learning Method {args.method}:")
            print(confusion_matrix(data["y_test"], y_pred))
            print_metrics(args.method, data, y_pred)
            print_prediction_result(data, y_pred)
            
    else: # trainandtest
        if not is_dataset_source(args.source):
                print(f"{args.source} is not dataset with extension .csv")
                sys.exit(1)

        if args.method in unsupervised:
            data = import_dataset(args.source, split=False)
            method = methods[args.method]
            y_pred = method(data)
            
            if args.method == "ocSVM" or "iF":
                for i, row in enumerate(y_pred):                    
                    if y_pred[i] == 1:
                        y_pred[i] = 0
                    else:
                        y_pred[i] = 1
                            
            # Print results
            print(f"Confusion Matrix of Machine Learning Method {args.method}:")
            print(confusion_matrix(data["y_test"], y_pred))
            print_metrics(args.method, data, y_pred)
            print_prediction_result(data, y_pred)
            # plot_graph(data)
        
        else: # supervised, deeplearning
            data = import_dataset(args.source, split=True)
            method = methods[args.method]
            classifier = method(data) 
            y_pred = classifier.predict(data["X_test"])
 
            if args.method in deepLearning:
                y_pred = (y_pred > 0.5)
                # Invert back to numbers
                y_pred = np.argmax(y_pred, axis = 1)
            
            # Print results
            print(f"Confusion Matrix of Machine Learning Method {args.method}:")
            print(confusion_matrix(data["y_test"], y_pred))
            print_metrics(args.method, data, y_pred)
            print_prediction_result(data, y_pred)

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
            classifier = load_classifier(args.source)
        output_filename = save_classifier(classifier, args.method)
        print(f"Trained classifier saved into file {output_filename}")
    
    elif args.command == "test": # test
        if not is_dataset_source(args.source):
            print(f"{args.source} is not dataset with extension .csv")
            sys.exit(1)
    
        data = import_unlabelled_dataset(args.source) # TODO New import
        if args.method in unsupervised:
            method = methods[args.method]
            y_pred = method(data)
            
            if args.method == "ocSVM" or "iF":
                for i, row in enumerate(y_pred):                    
                    if y_pred[i] == 1:
                        y_pred[i] = 0
                    else:
                        y_pred[i] = 1
                
        else: # supervised, deeplearning
            if args.method in deepLearning:
                classifier = load_classifier(f"classifiers/classifier_{args.method}.h5")
                y_pred = classifier.predict(data["X_test"])
                y_pred = (y_pred > 0.5)
                # Invert back to numbers
                y_pred = np.argmax(y_pred, axis = 1)
            else:
                classifier = load_classifier(f"classifiers/classifier_{args.method}.joblib")
                y_pred = classifier.predict(data["X_test"])
            
        labelled_dataset = np.c_[data["dataset"], ["Anomaly" if val else "Not anomaly" for val in y_pred]]
        np.set_printoptions(threshold=np.inf)
        with open(f"Results/{args.method}_labelled.csv", 'w') as f:
            for row in labelled_dataset:
                row = np.array(list(map(lambda s: s, row)))
                r = np.array2string(row, separator='\t ', max_line_width=np.inf, formatter={'str_kind': lambda x: x})
                f.write(f"{r[1:-1]}\n")
        print(f"Labelled dataset printed out to Results/{args.method}_labelled.csv")

    else: # trainandtest
        print("trainandtest is possible only in research mode")
        sys.exit(1)