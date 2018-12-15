import numpy as np 
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.initializers import glorot_uniform
from keras.regularizers import l2
from keras.models import model_from_json
from sklearn.model_selection import StratifiedKFold
from tensorboard import TrainValTensorBoard
from data import DataGenerator
import time

seed = 1997
kfold_splits = 5
batch_size = 32

def normalize_data(x):
    return 1 / (1 + np.exp(-x))



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

    model.add(LSTM(units=100, recurrent_dropout=0.2, dropout=0.2, input_shape=[17,20]))

    model.add(Dense(2, kernel_initializer=init, kernel_regularizer=reg))
    model.add(Activation('softmax'))

    tbCallBack = TrainValTensorBoard(log_dir='./lstm_balance/seed_{}/fold_{}/logs'.format(seed, index), histogram_freq=0, write_graph=True)
    ckptCallBack =  ModelCheckpoint(filepath='./lstm_balance/seed_{}/fold_{}/logs'.format(seed, index), monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
    model.summary()

    nb_epoch = 10
    model.fit_generator(train_gen,  
            epochs=nb_epoch,
            validation_data=(X_val_r, y_val), 
            callbacks=[tbCallBack, ckptCallBack], 
            class_weight={0: 0.4, 1: 0.6}
            )

    #metrics
    y_pred = model.predict(X_test_r)
    np.save('./lstm_balance/seed_{}/fold_{}/pred.npy'.format(seed, index), [y_test, y_pred])

    #save model
    model_json = model.to_json()
    with open("./lstm_balance/seed_{}/fold_{}/model.json".format(seed, index), "w") as json_file:
        json_file.write(model_json)
    model.save_weights("./lstm_balance/seed_{}/fold_{}/model.h5".format(seed, index))
    print("Saved model to disk")