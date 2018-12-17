from metrics import ultimate_metrics, ultimate_metrics_parse
import numpy as np 
import tensorflow as tf
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
from keras.backend import set_session
#config
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config) 
set_session(sess)


def normalize_data(x):
    return 1 / (1 + np.exp(-x))

#data test
test = np.loadtxt('fad.pssm.ws17.tst.csv', delimiter=',')
X_test_r = test[:, 1:].reshape(-1,17,20)
y_test_p = test[:, 0]
X_test_r = normalize_data(X_test_r)
y_test = to_categorical(y_test_p, num_classes=2)

k_f = 5
ms = ['lstm', 'bi', 'cnn']
seeds = [1,12,25,30,1997]

metrics = []   #m, seed, (k + ass), (sn, sp, acc, mcc, auc, fpr, tpr)

for m in ms:
  m_m = []
  for seed in seeds:
    m_s = []
    pred_assembles = []
    for k in range(k_f):
      #load model
      with open('{}_balance/seed_{}/fold_{}/model.json'.format(m, seed, k), 'r') as json_file:
        loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights("{}_balance/seed_{}/fold_{}/logs/best.h5".format(m, seed, k))

      #metrics
      y_pred = model.predict(X_test_r)
      pred_assembles.append(np.argmax(y_pred, axis=1))
      sn, sp, acc, mcc, auc, fpr, tpr = ultimate_metrics(y_test, y_pred)
      m_s.append([sn, sp, acc, mcc, auc, fpr[1], tpr[1]])
      print('>>>>>{}-{}-{}<<<<<'.format(m, seed, k))
      print ('sn: {}\nsp: {}\nacc: {}\nmcc: {}\nauc: {}\nfpr: {}\ntpr: {}'.format(sn, sp, acc, mcc, auc, fpr, tpr))
    y_pred_ass = []
    for r in np.array(pred_assembles).T.astype(np.int):
      y_pred_ass.append(np.bincount(r).argmax())
    sn, sp, acc, mcc, auc, fpr, tpr = ultimate_metrics_parse(y_test_p, np.array(y_pred_ass))
    m_s.append([sn, sp, acc, mcc, auc, fpr[1], tpr[1]])
    m_m.append(m_s)
  metrics.append(m_m)
metrics = np.array(metrics)
print(metrics.shape)
np.save('./metrics.npy', metrics)