"""
System Log Analysis for Anomaly Detection Using Machine Learning

@author: Miroslav Siklosi
"""

# Importing the libraries
import argparse
import sys
from joblib import dump, load

from data_preprocessing import import_dataset

parser = argparse.ArgumentParser(description="Train or test.")
parser.add_argument(
    "--action", dest="action", choices=["train", "test"], help="action - train or test"
)
parser.add_argument(
    "--method",
    dest="method",
    choices=["LR", "K-NN", "SVM", "kSVM", "NB", "DTC", "RFC", "K-Means", "HC", "ANN"],
    help="type of machine learning",
)
parser.add_argument(
    "--source-type",
    dest="source_type",
    choices=["dataset", "classifier"],
    help="type of source data",
)
parser.add_argument("--source", dest="source", help="source filename")
args = parser.parse_args()

supervised = ("LR", "K-NN", "SVM", "kSVM", "NB", "DTC", "RFC")
unsupervised = ("K-Means", "HC")
deepLearning = ("ANN")

methods = {"LR": method_LR, "K-NN": method_KNN, "SVM": method_SVM, "kSVM": method_kSVM,
           "NB": method_NB, "DTC": method_DTC, "K-Means": method_KMeans,
           "HC": method_HC, "ANN": method_ANN}

if args.action == "train":
    if args.method in unsupervised:
        print("Unsupervised does not need training...exiting")
        sys.exit()
    
    data = import_dataset(args.source)
    method = methods[args.method]
    knowledge = method(data)
    if args.method in supervised:
        output_filename = f"Data/classifier_{args.method}.joblib"
        dump(knowledge, output_filename)
    elif args.method in deepLearning:
        output_filename = f"Data/classifier_{args.method}.h5"
        knowledge.save(output_filename)
    print(f"Trained classifier save into file {output_filename}")
else: # test
    pass
    

data = import_dataset("Datasets/sample_data.csv")
classifier = method(data)

"""
Flag PROD or RESEARCH -> Change input variables (X and y) accordingly
in scripts ML_modules.py and predictions.py.

Output of PROD Mode - .csv file with X and y(predicted) merged
Output of RESEARCH Mode - Accuracies etc.


===============
Research Mode
===============
- Train and Test
	Choose ML method:
	- Supervised
		- LR
			- Import training dataset
			- Import classifier
		- K-NN
			- Import training dataset
			- Import classifier
		- SVM
			- Import training dataset
			- Import classifier
		- kSVM
			- Import training dataset
			- Import classifier
		- NB
			- Import training dataset
			- Import classifier
		- DTC
			- Import training dataset
			- Import classifier
		- RFC
			- Import training dataset
			- Import classifier
	- Unsupervised
		- K-Means
			- Import training dataset
			- Import classifier
		- HC
			- Import training dataset
			- Import classifier
	- Deep Learning
		- ANN
			- Import training dataset
			- Import classifier
Print Precision, Recall, F−measure and Confussion Matrix

- HELP
===============
Production Mode
===============
- Import Training Data
	Choose ML method:
	- Supervised
		- LR
			- Import training dataset
			- Import classifier
		- K-NN
			- Import training dataset
			- Import classifier
		- SVM
			- Import training dataset
			- Import classifier
		- kSVM
			- Import training dataset
			- Import classifier
		- NB
			- Import training dataset
			- Import classifier
		- DTC
			- Import training dataset
			- Import classifier
		- RFC
			- Import training dataset
			- Import classifier
	- Deep Learning
		- ANN
			- Import training dataset
			- Import classifier

- Import Test Data
	Choose ML method:
	- Supervised
		- LR (print labels in .csv file)
		- K-NN (print labels in .csv file)
		- SVM (print labels in .csv file)
		- kSVM (print labels in .csv file)
		- NB (print labels in .csv file)
		- DTC (print labels in .csv file)
		- RFC (print labels in .csv file)
	- Unsupervised (Import dataset with at least 10k lines)
		- K-Means (print labels in .csv file)
		- HC (print labels in .csv file)
	- Deep Learning
		- ANN (print labels in .csv file)
		
- HELP
"""






