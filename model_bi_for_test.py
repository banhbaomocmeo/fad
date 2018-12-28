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
from tensorboard import TrainValTensorBoard
from data import DataGenerator
from ultis import as_keras_metric, normalize_data
import time
from sklearn.model_selection import train_test_split

batch_size = 32
nb_epoch = 30


#config
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

#metric
auc = as_keras_metric(tf.metrics.auc)
recall = as_keras_metric(tf.metrics.recall)
precision = as_keras_metric(tf.metrics.precision)
f1_score = as_keras_metric(tf.contrib.metrics.f1_score)

#data
train = np.loadtxt('fad.pssm.ws17.trn.csv', delimiter=',')

X = train[:, 1:].reshape(-1,17,20)
y = train[:, 0]
X = normalize_data(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=None)

# Generate batches from indices
y_train, y_test = (to_categorical(y_train, num_classes=2), to_categorical(y_test, num_classes=2))

train_gen = DataGenerator(X_train, y_train, batch_size)

#build
model = Sequential()

model.add(Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.15, dropout=0.25), input_shape=[17,20]))
model.add(GlobalMaxPool1D())
model.add(Dense(100, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy', auc, recall, precision, f1_score])
model.summary()

tbCallBack = TrainValTensorBoard(log_dir='./bi_test/logs', histogram_freq=0, write_graph=True)
ckptCallBack =  ModelCheckpoint(filepath='./bi_test/best.h5', monitor='val_auc', verbose=1, save_best_only=True, mode='max')
reduceLRCallBack = ReduceLROnPlateau(monitor='val_f1_score', factor=0.3, patience=5, verbose=0, min_lr=1e-6)

model.fit_generator(train_gen,  
        epochs=nb_epoch,
        validation_data=(X_test, y_test), 
        callbacks=[tbCallBack, ckptCallBack, reduceLRCallBack], 
        #class_weight={0: 0.4, 1: 0.6}
        )

#save model
model_json = model.to_json()
with open("./bi_test/model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("./bi_test/model.h5")
print("Saved model to disk")