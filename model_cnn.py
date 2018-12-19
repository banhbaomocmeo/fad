import numpy as np 
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPooling1D, Conv1D
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix
from keras.initializers import glorot_uniform
from keras.regularizers import l2
from keras.models import model_from_json
from sklearn.model_selection import StratifiedKFold
from tensorboard import TrainValTensorBoard
from keras.callbacks import ModelCheckpoint
from ultis import as_keras_metric, normalize_data
from metrics import ultimate_metrics

seed = 1997
kfold_splits = 2


#config
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

#data
skf = StratifiedKFold(n_splits=kfold_splits, shuffle=True)

test = np.loadtxt('fad.pssm.ws17.tst.csv', delimiter=',')
train = np.loadtxt('fad.pssm.ws17.trn.csv', delimiter=',')

X = train[:, 1:].reshape(-1,17,20)
y = train[:, 0]
X_test_r = test[:, 1:].reshape(-1,17,20)
y_test = test[:, 0]

X = normalize_data(X)
X_test_r = normalize_data(X_test_r)
y_test = to_categorical(y_test, num_classes=2)

#metric
auc_roc = as_keras_metric(tf.metrics.auc)
recall = as_keras_metric(tf.metrics.recall)
precision = as_keras_metric(tf.metrics.precision)


for index, (train_indices, val_indices) in enumerate(skf.split(X, y)):
    print ("Training on fold " + str(index+1) + "/5...")

    # Generate batches from indices
    X_train_r, X_val_r = X[train_indices], X[val_indices]
    y_train, y_val = (to_categorical(y[train_indices], num_classes=2), to_categorical(y[val_indices], num_classes=2))
    

    #build
    model = Sequential()
    init = glorot_uniform(seed=seed)
    reg = l2(0.001)

    model.add(Conv1D(filters=32, kernel_size=5, padding='same', input_shape=(17,20), kernel_initializer=init, kernel_regularizer=reg))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2, strides=None, padding='same'))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', kernel_initializer=init, kernel_regularizer=reg))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2, strides=None, padding='same'))
    model.add(Conv1D(filters=128, kernel_size=3, padding='same', kernel_initializer=init, kernel_regularizer=reg))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dropout(0.4, seed=seed))
    model.add(Dense(128, kernel_initializer=init, kernel_regularizer=reg))
    model.add(Activation('relu'))
    model.add(Dropout(0.4, seed=seed))
    model.add(Dense(2, kernel_initializer=init, kernel_regularizer=reg))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=[auc_roc, recall, precision])
    print(model.metrics_names)
    nb_epoch = 10
    tbCallBack = TrainValTensorBoard(log_dir='./cnn/seed_{}/fold_{}/logs'.format(seed, index), histogram_freq=0, write_graph=True)
    ckptCallBack =  ModelCheckpoint(filepath='./lstm_balance/seed_{}/fold_{}/logs/best.h5'.format(seed, index), monitor='val_auc', verbose=1, save_best_only=True, mode='max')

    model.fit(X_train_r, y_train, 
            epochs=nb_epoch, 
            validation_data=(X_val_r, y_val), 
            batch_size=32, 
            callbacks=[tbCallBack, ckptCallBack], 
            class_weight = {0: 1, 1: 9})

    #metrics
    y_pred = model.predict(X_test_r)
    # np.save('./cnn/seed_{}/fold_{}/pred.npy'.format(seed, index), [y_test, y_pred])

    sn, sp, acc, mcc, auc, fpr, tpr = ultimate_metrics(y_test, y_pred)
    

    #save model
    model_json = model.to_json()
    with open("./cnn/seed_{}/fold_{}/model.json".format(seed, index), "w") as json_file:
        json_file.write(model_json)
    model.save_weights("./cnn/seed_{}/fold_{}/model.h5".format(seed, index))
    print("Saved model to disk")
    print('sn: {}\nsp: {}\nacc: {}\nmcc: {}\nauc: {}\nfpr: {}\ntpr: {}'.format(sn, sp, acc, mcc, auc, fpr, tpr))