import numpy as np 
import keras

# def generator(features, labels, batch_size):

#  # Create empty arrays to contain batch of features and labels#
#   pos = np.where(labels[:,1]==1)
#   neg = np.where(labels[:,1]==0)
#   pos_size = pos.shape[0]
#   neg_size = neg.shape[0]
#   half_bs = batch_size // 2

#   while True:
#     # choose random index in features
#     index_pos = np.random.choice(pos_size,half_bs)
#     index_neg = np.random.choice(neg_size,half_bs)
#     batch_feature_pos = features[pos][index_pos]
#     batch_feature_neg = features[neg][index_neg]
#     batch_label_pos = features[pos][index_pos]
#     batch_label_neg = features[neg][index_neg]
#     batch_features = np.vstack([batch_feature_pos, batch_feature_neg])
#     batch_labels = np.vstack([batch_label_pos, batch_label_neg])

#     yield batch_features, batch_labels



class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, features, labels, batch_size=32):
        'Initialization'
        self.features = features
        self.batch_size = batch_size
        self.labels = labels
        self.pos = np.where(self.labels[:,1]==1)[0]
        self.neg = np.where(self.labels[:,1]==0)[0]

        self.data_size = self.labels.shape[0]
        self.pos_size = self.pos.shape[0]
        self.neg_size = self.neg.shape[0]
        self.half_bs = self.batch_size // 2

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(self.neg_size // self.half_bs)

    def __getitem__(self, index):
        'Generate one batch of data'
        index_pos = self.pos[np.random.choice(self.pos_size,self.half_bs)]
        index_neg = self.neg[index*self.half_bs:(index+1)*self.half_bs]
        batch_feature_pos = self.features[index_pos]
        batch_feature_neg = self.features[index_neg]
        batch_label_pos = self.labels[index_pos]
        batch_label_neg = self.labels[index_neg]
        batch_features = np.vstack([batch_feature_pos, batch_feature_neg])
        batch_labels = np.vstack([batch_label_pos, batch_label_neg])

        return batch_features, batch_labels

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        indexes = np.arange(self.data_size)
        np.random.shuffle(indexes)
        self.features = self.features[indexes]
        self.labels = self.labels[indexes]
        self.pos = np.where(self.labels[:,1]==1)[0]
        self.neg = np.where(self.labels[:,1]==0)[0]
