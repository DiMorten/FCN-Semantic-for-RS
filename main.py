from utils import *
from keras.datasets import cifar10
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout
#from keras.callbacks import ModelCheckpoint , EarlyStopping
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as k
import keras

import numpy as np
from sklearn.utils import shuffle
import cv2
from skimage.util import view_as_windows
# Local

import deb


def extract_traning_patches():
    # Build function to extract pathches from the training image.
    return 0


def FCN(rows, cols, channels):
    # Define here the FCN model
    input_img = Input(shape=(rows, cols, channels))
    #...
    # output =
    return Model(input_img, output)


def Train(net, train_pathes_dir, batch_size,  epochs):

    print('Start the training')
    for epoch in range(epochs):
        loss_tr = np.zeros((1, 2))
        loss_ts = np.zeros((1, 2))
        # Loading the data set
        trn_data_dirs = glob.glob(train_pathes_dir + '/*.npy')
        # Random shuffle the data
        np.shuffle(trn_data_dirs)
        # Computing the number of batchs
        batch_idxs = len(trn_data_dirs) // batch_size

    for idx in xrange(0, batch_idxs):
        batch_files = trn_data_dirs[idx * batch_size:(idx + 1) * batch_size]
        batch = [load_data(batch_file) for batch_file in batch_files]
        batch_images = np.array(batch).astype(np.float32)
        x_train_b = batch[:, :, :, 0:3]
        y_train = batch[:, :, :, 3]

        # Performing hot encoding

        loss_tr = loss_tr + net.train_on_batch(x_train_b, y_train_hot_encoding)
    loss_tr = loss_tr / n_batchs_tr

    # # Evaluating the network in the test set

    # for  batch in range(n_batchs_ts):
    # 	x_test_b = x_test[batch * batch_size : (batch + 1) * batch_size , : , : , :]
    # 	y_test_h_b = y_test_h[batch * batch_size : (batch + 1) * batch_size , : ]
    #  loss_ts = loss_ts + net.test_on_batch(x_test_b , y_test_h_b)
    # loss_ts = loss_ts/n_batchs_ts

    # print("%d [training loss: %f , Train acc.: %.2f%%][Test loss: %f , Test acc.:%.2f%%]" %(epoch , loss_tr[0 , 0], 100*loss_tr[0 , 1] , loss_ts[0 , 0] , 100 * loss_ts[0 , 1]))


class Dataset(object):

    def __init__(self, patch_len=32, patch_step=32, path="../data/", im_name_train="Image_Train.tif", im_name_test="Image_Test.tif", label_name_train="Reference_Train.tif", label_name_test="Reference_Test.tif", channel_n=3, debug=1):
        self.patch_len = patch_len
        self.path = {"v": path, 'train': {}, 'test': {}}
        self.image = {'train': {}, 'test': {}}
        self.patches = {'train': {}, 'test': {}}
        self.path['train']['in'] = path + im_name_train
        self.path['test']['in'] = path + im_name_test
        self.path['train']['label'] = path + label_name_train
        self.path['test']['label'] = path + label_name_test
        self.channel_n = 3
        self.debug = debug
        self.patch_step=patch_step
    def create(self, patch_step=1):

        self.image["train"] = self.data_load(self.path['train'])
        self.image["test"] = self.data_load(self.path['test'])

        if self.debug:
            deb.prints(self.image["train"]['in'].shape)
            deb.prints(self.image["train"]['label'].shape)

            deb.prints(self.image["test"]['in'].shape)
            deb.prints(self.image["test"]['label'].shape)

        self.patches["train"] = self.patches_extract(
            self.image["train"])
        self.patches["test"] = self.patches_extract(
            self.image["test"])

    def data_load(self, path):
        image = {}
        image['in'] = cv2.imread(path['in'], -1)
        image['label'] = cv2.imread(path['label'], 0)
        return image

    def patches_extract(self, image):

        patches = {}
        patches['in'] = self.view_as_windows_multichannel(
            image['in'], (self.patch_len, self.patch_len, self.channel_n), step=self.patch_step)

        if self.debug: deb.prints(patches['in'].shape)

        return patches

    def view_as_windows_multichannel(self, arr_in, window_shape, step=1):
        out = np.squeeze(view_as_windows(arr_in, window_shape, step=step))
        out = np.reshape(out, (out.shape[0] * out.shape[1],) + out.shape[2::])
        return out

class Model(object):
	def __init__(self):
		self.metrics={}
		self.build()
	def build():
		

flag = {"data_create": True}
if __name__ == '__main__':
    #
    data = Dataset(patch_len=32)
    if flag['data_create']:
        data.create(patch_step=32)

    # extract_traning_patches()  # run this function once.
    # adam = Adam(lr=0.0001, beta_1=0.9)
    # net = FCN(rows, cols, channels)
    # net.summary()
    # net.compile(loss='binary_crossentropy',
    #             optimizer=adam, metrics=['accuracy'])
    # # Call the train function
    # Train(net, train_pathes_dir='dir', batch_size=100, epochs=50)
