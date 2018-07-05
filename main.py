from utils import *
from keras.datasets import cifar10
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, Conv2DTranspose
# from keras.callbacks import ModelCheckpoint , EarlyStopping
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
import keras

import numpy as np
from sklearn.utils import shuffle
import cv2
from skimage.util import view_as_windows
import argparse
import tensorflow as tf

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import metrics

from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,classification_report
# Local

from metrics import fmeasure,categorical_accuracy
import deb

parser = argparse.ArgumentParser(description='')
parser.add_argument('-pl', '--patch_len', dest='patch_len',
					type=int, default=32, help='patch len')
parser.add_argument('-pstr', '--patch_step_train', dest='patch_step_train',
					type=int, default=32, help='patch len')
parser.add_argument('-psts', '--patch_step_test', dest='patch_step_test',
					type=int, default=32, help='patch len')

parser.add_argument('-db', '--debug', dest='debug',
					type=int, default=1, help='patch len')
parser.add_argument('-ep', '--epochs', dest='epochs',
					type=int, default=100, help='patch len')
parser.add_argument('-bstr', '--batch_size_train', dest='batch_size_train',
					type=int, default=32, help='patch len')

parser.add_argument('-em', '--eval_mode', dest='eval_mode',
					default='metrics', help='Test evaluate mode: metrics or predict')
parser.add_argument('-is', '--im_store', dest='im_store',
					default=True, help='Store sample test predicted images')

args = parser.parse_args()


class NetObject(object):

	def __init__(self, patch_len=32, patch_step_train=32,patch_step_test=32, path="../data/", im_name_train="Image_Train.tif", im_name_test="Image_Test.tif", label_name_train="Reference_Train.tif", label_name_test="Reference_Test.tif", channel_n=3, debug=1, class_n=5):
		self.patch_len = patch_len
		self.path = {"v": path, 'train': {}, 'test': {}}
		self.image = {'train': {}, 'test': {}}
		self.patches = {'train': {}, 'test': {}}
		self.patches['train']['step']=patch_step_train
		self.patches['test']['step']=patch_len        
		self.path['train']['in'] = path + im_name_train
		self.path['test']['in'] = path + im_name_test
		self.path['train']['label'] = path + label_name_train
		self.path['test']['label'] = path + label_name_test
		self.channel_n = 3
		self.debug = debug
		self.class_n = 5



class Dataset(NetObject):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		if self.debug >= 1:
			print("Initializing Dataset instance")

	def create(self):
		self.image["train"], self.patches["train"] = self.subset_create(
			self.path['train'],self.patches["train"]['step'])
		self.image["test"], self.patches["test"] = self.subset_create(
			self.path['test'],self.patches["test"]['step'])

		if self.debug:
			deb.prints(self.image["train"]['in'].shape)
			deb.prints(self.image["train"]['label'].shape)

			deb.prints(self.image["test"]['in'].shape)
			deb.prints(self.image["test"]['label'].shape)

	def subset_create(self, path,patch_step):
		image = self.image_load(path)
		image['label_rgb']=image['label'].copy()
		image['label'] = self.label2idx(image['label'])
		patches = self.patches_extract(image,patch_step)
		return image, patches

	def image_load(self, path):
		image = {}
		image['in'] = cv2.imread(path['in'], -1)
		image['label'] = np.expand_dims(cv2.imread(path['label'], 0), axis=2)
		return image

	def patches_extract(self, image, patch_step):

		patches = {}
		patches['in'],_ = self.view_as_windows_multichannel(
			image['in'], (self.patch_len, self.patch_len, self.channel_n), step=patch_step)
		patches['label'],patches['label_partitioned_shape'] = self.view_as_windows_multichannel(
			image['label'], (self.patch_len, self.patch_len, 1), step=patch_step)
		#patches['label'] = np.expand_dims(patches['label'],axis=3)

		# ===================== Switch labels to one-hot ===============#
		##patches['label'] = np.reshape(patches['label'].shape[0],patches['label'].shape[1::])
		if self.debug >= 2:
			deb.prints(patches['label'].shape)

		if flag['label_one_hot']:

			# Get the vectorized integer label
			patches['label_h'] = np.reshape(
				patches['label'], (patches['label'].shape[0], patches['label'].shape[1]*patches['label'].shape[2]))
			deb.prints(patches['label_h'].shape)

			# Init the one-hot vectorized label
			patches['label_h2'] = np.zeros(
				(patches['label_h'].shape[0], patches['label_h'].shape[1], self.class_n))

			# Get the one-hot vectorized label
			for sample_idx in range(0, patches['label_h'].shape[0]):
				for loc_idx in range(0, patches['label_h'].shape[1]):
					patches['label_h2'][sample_idx, loc_idx,
										patches['label_h'][sample_idx][loc_idx]] = 1

				# deb.prints(np.squeeze(patches['label_h'][sample_idx].shape))
#				patches['label_h2'][sample_idx,:,np.squeeze(patches['label_h'][sample_idx])]=1

			# Get the image one-hot labels
			patches['label'] = np.reshape(patches['label_h2'], (patches['label_h2'].shape[0],
																patches['label'].shape[1], patches['label'].shape[2], self.class_n))

			if self.debug >= 2:
				deb.prints(patches['label_h2'].shape)

		# ============== End switch labels to one-hot =============#
		if self.debug:
			deb.prints(patches['label'].shape)
			deb.prints(patches['in'].shape)

		return patches

	def label2idx(self, image_label):
		unique = np.unique(image_label)
		idxs = np.array(range(0, unique.shape[0]))
		for val, idx in zip(unique, idxs):
			image_label[image_label == val] = idx
		return image_label

	def view_as_windows_multichannel(self, arr_in, window_shape, step=1):
		out = np.squeeze(view_as_windows(arr_in, window_shape, step=step))
		partitioned_shape=out.shape

		deb.prints(out.shape)
		out = np.reshape(out, (out.shape[0] * out.shape[1],) + out.shape[2::])
		return out,partitioned_shape

#=============== METRICS CALCULATION ====================#
	def ims_flatten(self,ims):
		return np.reshape(ims,(np.prod(ims.shape[0:-1]),ims.shape[-1])).astype(np.float64)

	def average_acc(self,y_pred,y_true):
		correct_per_class=np.zeros(self.class_n)
		correct_all=y_pred.argmax(axis=1)[y_pred.argmax(axis=1)==y_true.argmax(axis=1)]
		for clss in range(0,self.class_n):
			correct_per_class[clss]=correct_all[correct_all==clss].shape[0]
		if self.debug>=3:
			deb.prints(correct_per_class)

		_,per_class_count=np.unique(y_true.argmax(axis=1),return_counts=True)
		per_class_acc=np.divide(correct_per_class.astype('float32'),per_class_count.astype('float32'))
		average_acc=np.average(per_class_acc)
		return average_acc,per_class_acc
	def metrics_get(self,data): #requires batch['prediction'],batch['label']
		

		data['prediction_h'] = self.ims_flatten(data['prediction'])
		data['prediction_h']=self.probabilities_to_one_hot(data['prediction_h'])
				
		data['label_h'] = self.ims_flatten(data['label']) #(self.batch['test']['size']*self.patch_len*self.patch_len,self.class_n
		
		if self.debug>=3: 
			deb.prints(data['prediction_h'].dtype)
			deb.prints(data['label_h'].dtype)
			deb.prints(data['prediction_h'].shape)
			deb.prints(data['label_h'].shape)
			deb.prints(data['label_h'][0])
			deb.prints(data['prediction_h'][0])

		metrics={}
		metrics['f1_score']=f1_score(data['prediction_h'],data['label_h'],average='macro')
		metrics['overall_acc']=accuracy_score(data['prediction_h'],data['label_h'])
		
		
		metrics['confusion_matrix']=confusion_matrix(data['prediction_h'].argmax(axis=1),data['label_h'].argmax(axis=1))
		
		metrics['average_acc'],metrics['per_class_acc']=self.average_acc(data['prediction_h'],data['label_h'])

		if self.debug>=2: print(metrics['per_class_acc'])

		return metrics


	def probabilities_to_one_hot(self,vals):
		out=np.zeros_like(vals)
		out[np.arange(len(vals)), vals.argmax(1)] = 1
		return out

# =================== Image reconstruct =======================#

	def im_reconstruct(self,subset='test',mode='prediction'):
		h,w,_=self.image[subset]['label'].shape
		print(self.patches[subset]['label_partitioned_shape'])
		deb.prints(self.patches[subset][mode].shape)
		
		h_blocks,w_blocks,patch_len,_=self.patches[subset]['label_partitioned_shape']

		patches_block=np.reshape(self.patches[subset][mode].argmax(axis=3),(h_blocks,w_blocks,patch_len,patch_len))


		self.im_reconstructed=np.squeeze(np.zeros_like(self.image[subset]['label']))

		h_block_len=int(self.image[subset]['label'].shape[0]/h_blocks)
		w_block_len=int(self.image[subset]['label'].shape[1]/w_blocks)
		
		count=0
		# if self.debug>=2:
		# 	deb.prints(self.patches[subset]['prediction'][count].shape)
		# 	deb.prints(self.patches[subset]['prediction'].shape)
		# 	deb.prints(self.patches[subset]['label'].shape)
			
		for w_block in range(0,w_blocks):
			for h_block in range(0,h_blocks):
				y=int(h_block*h_block_len)
				x=int(w_block*w_block_len)
				#print(y)
				#print(x)				
				#deb.prints([y:y+self.patch_len])
				self.im_reconstructed[y:y+self.patch_len,x:x+self.patch_len]=patches_block[h_block,w_block,:,:]
				count+=1
		deb.prints(count)
		cv2.imwrite('../results/reconstructed/im_reconstructed_'+subset+'_'+mode+'.png',self.im_reconstructed.astype(np.uint8)*40)

		#print(h,w,h_blocks,w_blocks)
		#for 


#	def im_grayscale_idx_to_color(self,im):


def one_hot_multilabel_check(data):
	count_per_sample=np.count_nonzero(data,axis=1)
	unique,count=np.unique(count_per_sample,return_counts=True)
	print("unique,count",unique,count)



class NetModel(NetObject):
	def __init__(self, batch_size_train=32, batch_size_test=200, epochs=2, eval_mode='metrics', *args, **kwargs):
		super().__init__(*args, **kwargs)
		if self.debug >= 1:
			print("Initializing Model instance")
		self.metrics = {'train': {}, 'test': {}}
		self.batch = {'train': {}, 'test': {}}
		self.batch['train']['size'] = batch_size_train
		self.batch['test']['size'] = batch_size_test
		self.eval_mode = eval_mode
		self.epochs = epochs

	def transition_down(self, pipe, filters):
		pipe = Conv2D(filters, (3, 3), strides=(2, 2),
					  activation='relu', padding='same')(pipe)
		return pipe

	def dense_block(self, pipe, filters):
		pipe = Conv2D(filters, (3, 3), activation='relu', padding='same')(pipe)
		return pipe

	def transition_up(self, pipe, filters):
		pipe = Conv2DTranspose(filters, (3, 3), strides=(
			2, 2), activation='relu', padding='same')(pipe)
		return pipe

	def concatenate_transition_up(self, pipe1, pipe2, filters):
		pipe = keras.layers.concatenate([pipe1, pipe2], axis=3)
		
		#pipe = merge([pipe1, pipe2], mode='concat', concat_axis=3)
		pipe = self.transition_up(pipe, filters)
		return pipe

	def build(self):
		in_im = Input(shape=(self.patch_len, self.patch_len, self.channel_n))
		filters = 64

		pipe = {'down': [], 'up': []}
		c = {'down': 0, 'up': 0}

		pipe['down'].append(self.transition_down(in_im, filters))  # 0 16x16
		pipe['down'].append(self.transition_down(pipe['down'][0], filters*2))  # 1 8x8
		pipe['down'].append(self.transition_down(pipe['down'][1], filters*4))  # 2 4x4

		pipe['down'].append(self.dense_block(pipe['down'][2], filters*8))  # 3 4x4

		pipe['up'].append(self.concatenate_transition_up(pipe['down'][3], pipe['down'][2], filters*4))  # 0 8x8
		pipe['up'].append(self.concatenate_transition_up(pipe['up'][0], pipe['down'][1], filters*2))  # 1
		pipe['up'].append(self.concatenate_transition_up(pipe['up'][1], pipe['down'][0], filters))  # 2

		out = Conv2D(self.class_n, (1, 1), activation='softmax',
					 padding='same')(pipe['up'][-1])

		self.graph = Model(in_im, out)
		print(self.graph.summary())

	def compile(self, optimizer, loss='binary_crossentropy', metrics=['accuracy',metrics.categorical_accuracy]):
		self.graph.compile(loss=loss, optimizer=optimizer, metrics=metrics)

	def train(self, data):

		# Random shuffle
		data.patches['train']['in'], data.patches['train']['label'] = shuffle(data.patches['train']['in'], data.patches['train']['label'], random_state=0)
		#data.patches['test']['in'], data.patches['test']['label'] = shuffle(data.patches['test']['in'], data.patches['test']['label'], random_state=0)

		# Normalize
		data.patches['train']['in'] = normalize(data.patches['train']['in'].astype('float32'))
		data.patches['test']['in'] = normalize(data.patches['test']['in'].astype('float32'))

		# Computing the number of batches
		data.patches['train']['batch_n'] = data.patches['train']['in'].shape[0]//self.batch['train']['size']
		data.patches['test']['batch_n'] = data.patches['test']['in'].shape[0]//self.batch['test']['size']

		deb.prints(data.patches['train']['batch_n'])

		self.train_loop(data)


	def train_loop(self, data):
		print('Start the training')
		cback_tboard = keras.callbacks.TensorBoard(
			log_dir='../summaries/', histogram_freq=0, batch_size=self.batch['train']['size'], write_graph=True, write_grads=False, write_images=False)

		batch = {'train': {}, 'test': {}}
		self.batch['train']['n'] = data.patches['train']['in'].shape[0] // self.batch['train']['size']
		self.batch['test']['n'] = data.patches['test']['in'].shape[0] // self.batch['test']['size']

		data.patches['test']['prediction']=np.zeros_like(data.patches['test']['label'])
		count,unique=np.unique(data.patches['test']['label'].argmax(axis=3),return_counts=True)
		print("count,unique",count,unique)
		deb.prints(data.patches['test']['label'].shape)
		deb.prints(self.batch['test']['n'])
		
		#for epoch in [0,1]:
		for epoch in range(self.epochs):

			self.metrics['train']['loss'] = np.zeros((1, 4))
			self.metrics['test']['loss'] = np.zeros((1, 4))

			# Random shuffle the data
			data.patches['train']['in'], data.patches['train']['label'] = shuffle(data.patches['train']['in'], data.patches['train']['label'])

			for batch_id in range(0, self.batch['train']['n']):
			#for batch_id in range(0, 2):
				
				idx0 = batch_id*self.batch['train']['size']
				idx1 = (batch_id+1)*self.batch['train']['size']

				batch['train']['in'] = data.patches['train']['in'][idx0:idx1]
				batch['train']['label'] = data.patches['train']['label'][idx0:idx1]

				self.metrics['train']['loss'] += self.graph.train_on_batch(
					batch['train']['in'], batch['train']['label'])		# Accumulated epoch

			# Average epoch loss
			self.metrics['train']['loss'] /= self.batch['train']['n']


			for batch_id in range(0, self.batch['test']['n']):
				idx0 = batch_id*self.batch['test']['size']
				idx1 = (batch_id+1)*self.batch['test']['size']

				#deb.prints(data.patches['test']['label'].shape)
				#print(idx0,idx1)
				batch['test']['in'] = data.patches['test']['in'][idx0:idx1]
				batch['test']['label'] = data.patches['test']['label'][idx0:idx1]

				#deb.prints(batch['test']['label'].shape)
				self.metrics['test']['loss'] += self.graph.test_on_batch(
					batch['test']['in'], batch['test']['label'])		# Accumulated epoch

				#batch['test']['prediction']=self.graph.predict(batch['test']['in'],batch_size=self.batch['test']['size'])
				data.patches['test']['prediction'][idx0:idx1]=self.graph.predict(batch['test']['in'],batch_size=self.batch['test']['size'])

				if (batch_id % 4 == 0) and (epoch % 3 == 0):
					print("Saving image, batch id={}, epoch={}".format(batch_id,epoch))
					#print(data.patches['test']['prediction'][idx0].argmax(axis=2).astype(np.uint8)*50.shape)
					cv2.imwrite("../results/pred"+str(batch_id)+".png",data.patches['test']['prediction'][idx0].argmax(axis=2).astype(np.uint8)*50)
					cv2.imwrite("../results/label"+str(batch_id)+".png",data.patches['test']['label'][idx0].argmax(axis=2).astype(np.uint8)*50)
						
			
			print("Epoch={}".format(epoch))	
			
			# Get test metrics
			metrics=data.metrics_get(data.patches['test'])
			
			# Get, store test reconstructed image
			data.im_reconstruct(subset='test',mode='prediction')
			data.im_reconstruct(subset='test',mode='label')
			
			print('oa={}, aa={}, f1={}'.format(metrics['overall_acc'],metrics['average_acc'],metrics['f1_score']))
		
			# Average epoch loss
			self.metrics['test']['loss'] /= self.batch['test']['n']
			print("Train loss={}, Test loss={}".format(self.metrics['train']['loss'],self.metrics['test']['loss']))

			


flag = {"data_create": True, "label_one_hot": True}
if __name__ == '__main__':
	#
	data = Dataset(patch_len=args.patch_len, patch_step_train=args.patch_step_train)
	if flag['data_create']:
		data.create()

	adam = Adam(lr=0.0001, beta_1=0.9)
	model = NetModel(epochs=args.epochs, patch_len=args.patch_len,
					 patch_step_train=args.patch_step_train, eval_mode=args.eval_mode,batch_size_train=args.batch_size_train)
	model.build()
	model.compile(loss='binary_crossentropy',
				  optimizer=adam, metrics=['accuracy',fmeasure,categorical_accuracy])
	if args.debug:
		deb.prints(np.unique(data.patches['train']['label']))
		deb.prints(data.patches['train']['label'].shape)
	model.train(data)
