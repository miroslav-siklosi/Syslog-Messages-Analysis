"""
COMPARISON OF RESULTS

System Log Analysis for Anomaly Detection Using Machine Learning

@author: Miroslav Siklosi
"""
# Importing the libraries
import numpy as np
import data_preprocessing


''' Confusion Matrix '''
from sklearn.metrics import confusion_matrix

# CM of Classification models
CM_LR = confusion_matrix(data["y_test"], data["y_LR_pred"])
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
with open("Results/Confusion_Matrix_DTC.txt", 'w') as f:
            f.write(np.array2string(CM_DTC, separator=',', max_line_width=np.inf))
            
with open("Results/Confusion_Matrix_ANN.txt", 'w') as f:
            f.write(np.array2string(CM_ANN, separator=',', max_line_width=np.inf))
            
with open("Results/Confusion_Matrix_KMeans.txt", 'w') as f:
            f.write(np.array2string(CM_KMeans, separator=',', max_line_width=np.inf))

''' Matrix Diagonal Summary (TP + TN) '''
def matrix_Sum_Diag(matrix):
    SUM = 0
    for i, row in enumerate(matrix):
        SUM = SUM + matrix[i, i]
    return  SUM

''' Matrix Summary of everything except Diagonal (FP + FN) '''
def matrix_Sum_Except_Diag(matrix):
    SUM = 0
    for i, row in enumerate(CM_RFC):
        for j, column in enumerate(CM_RFC):
            if  (i != j):
                SUM = SUM + matrix[i, j]
    return SUM

''' False Negative of specific Label '''
def matrix_Label_FN(matrix, label):
    SUM = 0
    for i, row in enumerate(matrix):
        if  (i != label):
            SUM = SUM + matrix[label, i]
    return  SUM

''' False Positive of specific Label '''
def matrix_Label_FP(matrix, label):
    SUM = 0
    for i, row in enumerate(matrix):
        if  (i != label):
            SUM = SUM + matrix[i, label]
    return  SUM

''' True Positive of specific Label '''
def matrix_Label_TP(matrix, label):
    return matrix[label, label]

''' True Negative of specific Label '''
def matrix_Label_TN(matrix, label):
    return matrix_Sum_Diag(matrix) - matrix_Label_TP(matrix, label)

''' Precision of specific Label '''
def matrix_Label_Precision(matrix, label):
    precision = 100 * ((matrix_Label_TP(matrix, label)) /
                 (matrix_Label_TP(matrix, label) + matrix_Label_FP(matrix, label)))
    return precision

''' Sensitivity of specific Label '''
def matrix_Label_Sensitivity(matrix, label):
    sensitivity = 100 * ((matrix_Label_TP(matrix, label)) /
    (matrix_Label_TP(matrix, label) + matrix_Label_FN(matrix, label)))
    return sensitivity

''' Specificity of specific Label '''
def matrix_Label_Specificity(matrix, label):
    specificity = 100 * ((matrix_Label_TN(matrix, label)) /
                   (matrix_Label_FP(matrix, label) + matrix_Label_TN(matrix, label)))
    return specificity

''' F-Score of specific Label '''
def matrix_Label_Fscore(matrix, label):
    Fscore = 100 * (2 * (matrix_Label_TP(matrix, label)) /
              (2 * (matrix_Label_TP(matrix, label)) +
               matrix_Label_FP(matrix, label) + matrix_Label_FN(matrix, label)))
    return Fscore


def matrix_Label_Results(matrix, label):
    precision = matrix_Label_Precision(matrix, label)
    sensitivity = matrix_Label_Sensitivity(matrix, label)
    specificity = matrix_Label_Specificity(matrix, label)
    Fscore = matrix_Label_Fscore(matrix, label)
    return {"Precision": precision, "Sensitivity": sensitivity,
              "Specificity": specificity, "F-Score": Fscore}

print('Results of model DTC and Label 0 are', matrix_Label_Results(CM_DTC, 0))
print('Results of model ANN and Label 0 are', matrix_Label_Results(CM_ANN, 0))