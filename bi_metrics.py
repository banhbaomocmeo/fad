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


metrics = []   #m, seed, (k + ass), (sn, sp, acc, mcc, auc, fpr, tpr)


with open('bi_test/model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights("bi_test/best.h5")

#metrics
y_pred = model.predict(X_test_r)
sn, sp, acc, mcc, auc, fpr, tpr = ultimate_metrics(y_test, y_pred)
print ('sn: {}\nsp: {}\nacc: {}\nmcc: {}\nauc: {}\nfpr: {}\ntpr: {}'.format(sn, sp, acc, mcc, auc, fpr, tpr))
