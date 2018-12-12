import numpy as np 
from sklearn import metrics

def sensitivity(mat):
    tn, fp, fn, tp = mat
    return tp / (tp + fn)

def specificity(mat):
    tn, fp, fn, tp = mat
    return tn / (tn + fp)

def accuracy(mat):
    tn, fp, fn, tp = mat
    return (tp + tn) / (tp + fp + tn + fn)

def matthews_correlation_coefficient(mat):
    tn, fp, fn, tp = mat
    return (tp * tn - fp * fn) / np.sqrt((tp + fn)*(tp + fp)*(tn + fp)*(tn + fn))

def auc_score(y_truth, y_pred):
    auc = metrics.roc_auc_score(y_truth, y_pred)
    return auc

def roc(y_truth, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_truth, y_pred, pos_label=1)
    return fpr, tpr

def ultimate_metrics(y_truth, y_pred):
    y_parse = np.argmax(y_truth, axis=1)
    y_round = np.argmax(y_pred, axis=1)
    
    conf_mat = metrics.confusion_matrix(y_parse, y_round).ravel() #tn, fp, fn, tp
    sn = sensitivity(conf_mat)
    sp = specificity(conf_mat)
    acc = accuracy(conf_mat)
    mcc = matthews_correlation_coefficient(conf_mat)
    auc = auc_score(y_truth, y_pred)
    fpr, tpr = roc(y_parse, y_round)
    return sn, sp, acc, mcc, auc, fpr, tpr