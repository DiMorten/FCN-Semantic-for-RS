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
parser.add_argument('-db', '--debug', dest='debug',
					type=int, default=1, help='patch len')

args = parser.parse_args()


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
		image['label'] = np.expand_dims(cv2.imread(path['label'], 0),axis=2)
		return image

	def patches_extract(self, image):

		patches = {}
		patches['in'] = self.view_as_windows_multichannel(image['in'], (self.patch_len, self.patch_len, self.channel_n), step=self.patch_step)
		patches['label'] = self.view_as_windows_multichannel(image['label'], (self.patch_len, self.patch_len, 1), step=self.patch_step)
		patches['label'] = np.expand_dims(patches['label'],axis=3)
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
	def __init__(self, batch_size=1,batch_size_test=10, epochs=2, *args, **kwargs):
		super().__init__(*args, **kwargs)
		if self.debug>=1: print("Initializing Model instance")
		self.metrics={'train':{},'test':{}}
		self.batch={'train':{},'test':{}}
		self.batch['train']['size']=batch_size
		self.batch['test']['size']=batch_size_test
		
		self.epochs=epochs
		self.build()
		
	def build(self):
		in_im = Input(shape=(self.patch_len, self.patch_len, self.channel_n))
		out = Conv2D(1 , (5 , 5) , activation='relu' , padding='same')(in_im)
		self.graph = Model(in_im,out)
		
	def compile(self,optimizer,loss='binary_crossentropy',metrics=['accuracy']):
		self.graph.compile(loss=loss,optimizer=optimizer,metrics=metrics)

	def train(self,data):
		#Random shuffle 	
		data['train']['in'],data['train']['label'] = shuffle(data['train']['in'],data['train']['label'], random_state = 0)
		data['test']['in'],data['test']['label'] = shuffle(data['test']['in'],data['test']['label'], random_state = 0)

		#Normalize
		data['train']['in'] = normalize(data['train']['in'].astype('float32'))
		data['test']['in'] = normalize(data['test']['in'].astype('float32'))
		
		#Computing the number of batches
		data['train']['batch_n'] = data['train']['in'].shape[0]//self.batch['train']['size']
		data['test']['batch_n'] = data['test']['in'].shape[0]//self.batch['test']['size']

		deb.prints(data['train']['batch_n'])

		self.train_loop(data)


	def train_loop(self,data):
		print('Start the training')
		
		batch={'train':{},'test':{}}
		batch['train']['n'] = data['train']['in'].shape[0] // self.batch['train']['size']
		batch['test']['n'] = data['test']['in'].shape[0] // self.batch['test']['size']
		
		for epoch in range(self.epochs):

			self.metrics['train']['loss'] = np.zeros((1, 2))
			self.metrics['test']['loss'] = np.zeros((1, 2))
			
			# Random shuffle the data
			data['train']['in'],data['train']['label']=shuffle(data['train']['in'],data['train']['label'])
			
			for batch_id in range(0, batch['train']['n']):
				idx0 = batch_id*self.batch['train']['size']
				idx1 = (batch_id+1)*self.batch['train']['size']

				batch['train']['in']=data['train']['in'][idx0:idx1]
				batch['train']['label']=data['train']['label'][idx0:idx1]
				
				self.metrics['train']['loss']+=self.graph.train_on_batch(batch['train']['in'],batch['train']['label'])		# Accumulated epoch
			

			self.metrics['train']['loss']/=self.batch['train']['n']		# Average epoch loss
			deb.prints(self.metrics['train']['loss'])
	

			for batch_id in range(0, batch['test']['n']):
				idx0 = batch_id*self.batch['test']['size']
				idx1 = (batch_id+1)*self.batch['test']['size']

				batch['test']['in']=data['test']['in'][idx0:idx1]
				batch['test']['label']=data['test']['label'][idx0:idx1]
				
				self.metrics['test']['loss']+=self.graph.test_on_batch(batch['test']['in'],batch['test']['label'])		# Accumulated epoch
			

			self.metrics['test']['loss']/=self.batch['test']['n']		# Average epoch loss
			deb.prints(self.metrics['test']['loss'])
	

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
	if args.debug:
		deb.prints(np.unique(data.patches['train']['label']))
		deb.prints(data.patches['train']['label'].shape)
	model.train(data.patches)
	# extract_traning_patches()  # run this function once.
	# net = FCN(rows, cols, channels)
	# net.summary()
	# net.compile(loss='binary_crossentropy',
	#             optimizer=adam, metrics=['accuracy'])
	# # Call the train function
	# Train(net, train_pathes_dir='dir', batch_size=100, epochs=50)
