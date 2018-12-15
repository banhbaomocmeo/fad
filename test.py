import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, Bidirectional, GlobalMaxPool1D
from keras.optimizers import Adam



model = Sequential()


model.add(Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1, dropout=0.25), input_shape=[17,20]))
model.add(GlobalMaxPool1D())
model.add(Dense(100, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
model.summary()