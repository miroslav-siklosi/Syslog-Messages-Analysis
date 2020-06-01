# System Log Analysis for Anomaly Detection Using Machine Learning
### Tool for security anomaly detection and analysis using Machine Leaning methods, created as part of Master Thesis.

Syslog Analysis tool is analyzing syslog messages created by [Syslog Generator](https://github.com/UsernameIsNotTakenAtAll/Syslog-Generator). This tool is created to detect security anomalies using Machine Learning methods.

Tool can be run by command from the tool's folder. There are no default arguments, so everything needs to be specified.

```
python syslog_messages_analysis.py --mode <> --model <> --command <> --source <>
```

There are four arguments and all of them are mandatory. The options are following:

- --mode *<research/prod>*
- --model *<LR/K-NN/kSVM/NB/DTC/RFC/ocSVM/iF/LOF/K-Means/HC/ANN>*
- --command *<train/predict/trainandpredict>*
- --source <*filename*>

Argument *mode* specifies in which mode the tool should run. Options are research(*research*) and production(*prod*). Each mode differentiate in output it return. Research mode returns metrics of predictions of machine learning models, such as confusion matrix, accuracy, precision, recall and F1-Score. Production mode labels imported messages whether they are anomaly (*1*) or not (*0*). Labelled dataset is then saved into file text file in folder *Results*.

Argument *model* chooses which machine learning model to use. Options are Logistic Regression(*LR*), K-Nearest Neighbors(*K-NN*), Kernel SVM(*kSVM*), Naive Bayes(*NB*), Decision Tree Classifier(*DTC*), Random Forest Classificier(*RFC*), One-class SVM(*ocSVM*), Isolation Forest(*iF*), Local Outlier Factor(*LOF*), K-Means(K-*Means*), Hierarchical Classifier(*HC*) and Artoficial Neural Network(*ANN*).

Argument *command* chooses what action should the tool do. Option are to train the machine learning model (*train*), predict anomalies based on learned weights (*predict*) or train and predict machine learning model on the same dataset (*trainandpredict*). Train and predict is specific command usable only in mode *research*. Using it in mode *prod* will return an error.

Last but not least argument is *source*. This argument specifies which file (dataset) should be imported into tool for training or predictions. If the file is not in same folder as the tool, full filepath needs to be specified.

### Author
- Miroslav Siklosi

### License
This project is licensed under the MIT License - see the LICENSE.md file for details.
