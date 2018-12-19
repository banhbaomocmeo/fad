import numpy as np 
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, Bidirectional, GlobalMaxPool1D
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.initializers import glorot_uniform
from keras.regularizers import l2
from keras.models import model_from_json
from sklearn.model_selection import StratifiedKFold
from tensorboard import TrainValTensorBoard
from data import DataGenerator
from ultis import as_keras_metric, normalize_data
import time

kfold_splits = 5
batch_size = 32
seeds = [25,12,30,1,1997]


#config
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

#metric
auc = as_keras_metric(tf.metrics.auc)
recall = as_keras_metric(tf.metrics.recall)
precision = as_keras_metric(tf.metrics.precision)

#data
skf = StratifiedKFold(n_splits=kfold_splits, shuffle=True)

train = np.loadtxt('fad.pssm.ws17.trn.csv', delimiter=',')

X = train[:, 1:].reshape(-1,17,20)
y = train[:, 0]

X = normalize_data(X)

for seed in seeds:

    for index, (train_indices, val_indices) in enumerate(skf.split(X, y)):
        print ("Training on fold " + str(index+1) + "/5...")

        # Generate batches from indices
        X_train_r, X_val_r = X[train_indices], X[val_indices]
        y_train, y_val = (to_categorical(y[train_indices], num_classes=2), to_categorical(y[val_indices], num_classes=2))
        
        train_gen = DataGenerator(X_train_r, y_train, batch_size)

        #build
        model = Sequential()
        init = glorot_uniform(seed=seed)
        reg = l2(0.001)

        model.add(Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1, dropout=0.25, kernel_initializer=init), input_shape=[17,20]))
        model.add(GlobalMaxPool1D())
        model.add(Dense(100, activation="relu", kernel_initializer=init, kernel_regularizer=reg))
        model.add(Dropout(0.25, seed=seed))
        model.add(Dense(2, kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy', auc, recall, precision])
        model.summary()

        tbCallBack = TrainValTensorBoard(log_dir='./bi_balance/seed_{}/fold_{}/logs'.format(seed, index), histogram_freq=0, write_graph=True)
        ckptCallBack =  ModelCheckpoint(filepath='./bi_balance/seed_{}/fold_{}/logs/best.h5'.format(seed, index), monitor='val_auc', verbose=1, save_best_only=True, mode='max')
        reduceLRCallBack = ReduceLROnPlateau(monitor='val_auc', factor=0.3, patience=5, verbose=0, min_lr=1e-6)
        
        nb_epoch = 30
        model.fit_generator(train_gen,  
                epochs=nb_epoch,
                validation_data=(X_val_r, y_val), 
                callbacks=[tbCallBack, ckptCallBack, reduceLRCallBack], 
                class_weight={0: 0.4, 1: 0.6}
                )

        #save model
        model_json = model.to_json()
        with open("./bi_balance/seed_{}/fold_{}/model.json".format(seed, index), "w") as json_file:
            json_file.write(model_json)
        model.save_weights("./bi_balance/seed_{}/fold_{}/model.h5".format(seed, index))
        print("Saved model to disk")