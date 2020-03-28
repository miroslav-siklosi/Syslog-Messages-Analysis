"""
COMPARISON OF RESULTS

System Log Analysis for Anomaly Detection Using Machine Learning

@author: Miroslav Siklosi
"""
# Importing the libraries
import numpy as np
import data_preprocessing

''' Confusion Matrix '''
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

# CM of Classification models
CM_LR = confusion_matrix(y_test, y_LR_pred)
CM_KNN = confusion_matrix(y_test, y_KNN_pred)
CM_SVM = confusion_matrix(y_test, y_SVM_pred)
CM_kSVM = confusion_matrix(y_test, y_kSVM_pred)
CM_NB = confusion_matrix(y_test, y_NB_pred)
CM_DTC = confusion_matrix(y_test, y_DTC_pred) 
CM_RFC = confusion_matrix(y_test, y_RFC_pred)

# CM of Clustering models
CM_KMeans = confusion_matrix(y, y_kmeans)
CM_HC = confusion_matrix(y, y_hc)

# CM of Neural Network
CM_ANN = confusion_matrix(y_test, y_ANN_pred)

# Print CMs into text file
with open("Confusion_Matrix_DTC.txt", 'w') as f:
            f.write(np.array2string(CM_DTC, separator=',', max_line_width=np.inf))
            
with open("Confusion_Matrix_ANN.txt", 'w') as f:
            f.write(np.array2string(CM_ANN, separator=',', max_line_width=np.inf))
            
with open("Confusion_Matrix_KMeans.txt", 'w') as f:
            f.write(np.array2string(CM_KMeans, separator=',', max_line_width=np.inf))
