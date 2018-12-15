from metrics import ultimate_metrics 
import numpy as np 

y = np.load('lstm/pred.npy')
# print(y)
# print(y.shape)
# print(max(y[1]), min(y[1]))
# print(y[0][np.where(y[0]==1)].shape[0])
# print(y[0][np.where(y[0]==0)].shape[0])
sn, sp, acc, mcc, auc, fpr, tpr = ultimate_metrics(y[0], y[1])
print ('sn: {}\nsp: {}\nacc: {}\nmcc: {}\nauc: {}\nfpr: {}\ntpr: {}'.format(sn, sp, acc, mcc, auc, fpr, tpr))