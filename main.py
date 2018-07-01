from utils import *
from keras.datasets import cifar10
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout
# from keras.callbacks import ModelCheckpoint , EarlyStopping
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as k
import keras

import numpy as np
from sklearn.utils import shuffle
import cv2
from skimage.util import view_as_windows
import argparse

# Local

import deb

parser = argparse.ArgumentParser(description='')
parser.add_argument('-pl', '--patch_len', dest='patch_len',
					type=int, default=32, help='patch len')
parser.add_argument('-ps', '--patch_step', dest='patch_step',
					type=int, default=32, help='patch len')
args = parser.parse_args()


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

	# print("%d [training loss: %f , Train acc.: %.2f%%][Test loss: %f , Test
	# acc.:%.2f%%]" %(epoch , loss_tr[0 , 0], 100*loss_tr[0 , 1] , loss_ts[0 ,
	# 0] , 100 * loss_ts[0 , 1]))


class NetObject(object):

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
		self.patch_step = patch_step


class Dataset(NetObject):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		if self.debug >= 1: print("Initializing Dataset instance")

	def create(self):
		self.image["train"],self.patches["train"]=self.subset_create(self.path['train'])
		self.image["test"],self.patches["test"]=self.subset_create(self.path['test'])

		if self.debug:
			deb.prints(self.image["train"]['in'].shape)
			deb.prints(self.image["train"]['label'].shape)

			deb.prints(self.image["test"]['in'].shape)
			deb.prints(self.image["test"]['label'].shape)

	def subset_create(self, path):
		image = self.image_load(path)
		image['label'] = self.label2idx(image['label'])
		patches = self.patches_extract(image)
		return image,patches

	def image_load(self, path):
		image = {}
		image['in'] = cv2.imread(path['in'], -1)
		image['label'] = cv2.imread(path['label'], 0)
		return image

	def patches_extract(self, image):

		patches = {}
		patches['in'] = self.view_as_windows_multichannel(image['in'], (self.patch_len, self.patch_len, self.channel_n), step=self.patch_step)
		patches['label'] = self.view_as_windows_multichannel(image['label'], (self.patch_len, self.patch_len), step=self.patch_step)

		if self.debug: deb.prints(patches['in'].shape)

		return patches
	def label2idx(self,image_label):
		unique=np.unique(image_label)
		idxs=np.array(range(0,unique.shape[0]))
		for val,idx in zip(unique,idxs):
			image_label[image_label==val]=idx
		return image_label
	def view_as_windows_multichannel(self, arr_in, window_shape, step=1):
		out = np.squeeze(view_as_windows(arr_in, window_shape, step=step))
		out = np.reshape(out, (out.shape[0] * out.shape[1],) + out.shape[2::])
		return out

class NetModel(NetObject):
	def __init__(self, batch_size=1, *args, **kwargs):
		super().__init__(*args, **kwargs)
		if self.debug>=1: print("Initializing Model instance")
		self.metrics={}
		self.batch_size=batch_size
		self.build()
		
	def build(self):
		in_im = Input(shape=(self.patch_len, self.patch_len, self.channel_n))
		out = Conv2D(32 , (5 , 5) , activation='relu' , padding='same')(in_im)
		self.graph = Model(in_im,out)
		
	def compile(self,optimizer,loss='binary_crossentropy',metrics=['accuracy']):
		self.graph.compile(loss=loss,optimizer=optimizer,metrics=metrics)

	def train(self,data):
		#Random shuffle the data		
		data['train']['in'],data['train']['label'] = shuffle(data['train']['in'],data['train']['label'], random_state = 0)
		data['test']['in'],data['test']['label'] = shuffle(data['test']['in'],data['test']['label'], random_state = 0)

		#Normalizing the set
		data['train']['in'] = normalize(data['train']['in'].astype('float32'))
		data['test']['in'] = normalize(data['test']['in'].astype('float32'))
		
		#Computing the number of batchs
		data['train']['batch_n'] = data['train']['in'].shape[0]//self.batch_size
		data['test']['batch_n'] = data['test']['in'].shape[0]//self.batch_size
		
		deb.prints(data['train']['batch_n'])

		#data['train']['label'] = normalize(data['train']['label'])


flag = {"data_create": True}
if __name__ == '__main__':
	#
	data = Dataset(patch_len=args.patch_len,patch_step=args.patch_step)
	if flag['data_create']:
		data.create()

	adam = Adam(lr=0.0001, beta_1=0.9)
	model = NetModel(patch_len=args.patch_len,patch_step=args.patch_step)
	model.build()
	model.compile(loss='binary_crossentropy',optimizer=adam, metrics=['accuracy'])
	deb.prints(np.unique(data.patches['train']['label']))
	model.train(data.patches)
	# extract_traning_patches()  # run this function once.
	# net = FCN(rows, cols, channels)
	# net.summary()
	# net.compile(loss='binary_crossentropy',
	#             optimizer=adam, metrics=['accuracy'])
	# # Call the train function
	# Train(net, train_pathes_dir='dir', batch_size=100, epochs=50)
