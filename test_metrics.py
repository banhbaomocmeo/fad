from metrics import ultimate_metrics 
import numpy as np 

ms = ['lstm', 'bi', 'cnn']

for m in ms:

  y = np.load('{}_balance/seed_1997/fold_4/pred.npy'.format(m))
# print(y)
# print(y.shape)
# print(max(y[1]), min(y[1]))
# print(y[0][np.where(y[0]==1)].shape[0])
# print(y[0][np.where(y[0]==0)].shape[0])
  sn, sp, acc, mcc, auc, fpr, tpr = ultimate_metrics(y[0], y[1])
  print('>>>>>{}<<<<<'.format(m))
  print ('sn: {}\nsp: {}\nacc: {}\nmcc: {}\nauc: {}\nfpr: {}\ntpr: {}'.format(sn, sp, acc, mcc, auc, fpr, tpr))
