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
from keras_weighted_categorical_crossentropy import weighted_categorical_crossentropy

parser = argparse.ArgumentParser(description='')
parser.add_argument('-pl', '--patch_len', dest='patch_len',
					type=int, default=32, help='patch len')
parser.add_argument('-pstr', '--patch_step_train', dest='patch_step_train',
					type=int, default=32, help='patch len')
parser.add_argument('-psts', '--patch_step_test', dest='patch_step_test',
					type=int, default=None, help='patch len')

parser.add_argument('-db', '--debug', dest='debug',
					type=int, default=1, help='patch len')
parser.add_argument('-ep', '--epochs', dest='epochs',
					type=int, default=30, help='patch len')
parser.add_argument('-pt', '--patience', dest='patience',
					type=int, default=30, help='patience')

parser.add_argument('-bstr', '--batch_size_train', dest='batch_size_train',
					type=int, default=32, help='patch len')
parser.add_argument('-bsts', '--batch_size_test', dest='batch_size_test',
					type=int, default=32, help='patch len')

parser.add_argument('-em', '--eval_mode', dest='eval_mode',
					default='metrics', help='Test evaluate mode: metrics or predict')
parser.add_argument('-is', '--im_store', dest='im_store',
					default=True, help='Store sample test predicted images')
parser.add_argument('-eid', '--exp_id', dest='exp_id',
					default='default', help='Experiment id')

args = parser.parse_args()

if args.patch_step_test==None:
	args.patch_step_test=args.patch_len

deb.prints(args.patch_step_test)

class NetObject(object):

	def __init__(self, patch_len=32, patch_step_train=32,patch_step_test=32, path="../data/", im_name_train="Image_Train.tif", im_name_test="Image_Test.tif", label_name_train="Reference_Train.tif", label_name_test="Reference_Test.tif", channel_n=3, debug=1, class_n=5,exp_id="skip_connections"):
		self.patch_len = patch_len
		self.path = {"v": path, 'train': {}, 'test': {}}
		self.image = {'train': {}, 'test': {}}
		self.patches = {'train': {}, 'test': {}}
		self.patches['train']['step']=patch_step_train
		self.patches['test']['step']=patch_step_test        
		self.path['train']['in'] = path + im_name_train
		self.path['test']['in'] = path + im_name_test
		self.path['train']['label'] = path + label_name_train
		self.path['test']['label'] = path + label_name_test
		self.channel_n = 3
		self.debug = debug
		self.class_n = 5
		self.report={'best':{}}
		self.report['exp_id']=exp_id
		self.report['best']['text_name']='result_'+exp_id+'.txt'
		self.report['best']['text_path']='../results/'+self.report['best']['text_name']
		
		#self.report['best']['im_reconstruct_predict_name']='im_reconstruct_predict_'+exp_id+'.png'






class Dataset(NetObject):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.im_gray_idx_to_rgb_table=[[0,[0,0,255],29],
									[1,[0,255,0],150],
									[2,[0,255,255],179],
									[3,[255,255,0],226],
									[4,[255,255,255],255]]
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
		count,unique=np.unique(image['label'],return_counts=True)
		print("label count,unique",count,unique)
		image['label_rgb']=cv2.imread(path['label'], -1)
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
		if self.debug >= 0:
			deb.prints(patches['label'].shape)
			cv2.imwrite('../results/label_patch_sample0.png',patches['label'][0].astype(np.uint8)*100)
			cv2.imwrite('../results/label_patch_sample19.png',patches['label'][19].astype(np.uint8)*100)


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
	def flattened_to_im(self,data_h,im_shape):
		return np.reshape(data_h,im_shape)

	def probabilities_to_one_hot(self,vals):
		out=np.zeros_like(vals)
		out[np.arange(len(vals)), vals.argmax(1)] = 1
		return out
	def assert_equal(self,val1,val2):
		return np.equal(val1,val2)


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

		data_label_reconstructed=self.flattened_to_im(data['label_h'],data['label'].shape)
		data_prediction_reconstructed=self.flattened_to_im(data['prediction_h'],data['label'].shape)
		
		deb.prints(data_label_reconstructed.shape)
		np.testing.assert_almost_equal(data['label'],data_label_reconstructed)
		print("Is label reconstructed equal to original",np.array_equal(data['label'],data_label_reconstructed))
		print("Is prediction reconstructed equal to original",np.array_equal(data['prediction'].argmax(axis=3),data_prediction_reconstructed.argmax(axis=3)))


		
		#np.assert_equal(data['label'],data_label_reconstructed)

		if self.debug>=2: print(metrics['per_class_acc'])

		return metrics

	def metrics_write_to_txt(self,metrics,epoch=0):
		with open(self.report['best']['text_path'], "w") as text_file:
		    text_file.write("Overall_acc,average_acc,f1_score: {0},{1},{2},{3}".format(str(metrics['overall_acc']),str(metrics['average_acc']),str(metrics['f1_score']),str(epoch)))
		


# =================== Image reconstruct =======================#

	def im_reconstruct(self,subset='test',mode='prediction'):
		h,w,_=self.image[subset]['label'].shape
		print(self.patches[subset]['label_partitioned_shape'])

		#h=h-30 # Last 30 vertical pixels were not taken into account
		deb.prints(self.patches[subset][mode].shape)
		
		h_blocks,w_blocks,patch_len,_=self.patches[subset]['label_partitioned_shape']

		patches_block=np.reshape(self.patches[subset][mode].argmax(axis=3),(h_blocks,w_blocks,patch_len,patch_len))


		self.im_reconstructed=np.squeeze(np.zeros_like(self.image[subset]['label']))

		#h_block_len=int(self.image[subset]['label'].shape[0]/h_blocks)
		#w_block_len=int(self.image[subset]['label'].shape[1]/w_blocks)
		w_block_len=self.patch_len
		h_block_len=self.patch_len
		
		deb.prints(h_block_len)
		deb.prints(w_block_len)
		
		count=0

		for w_block in range(0,w_blocks):
			for h_block in range(0,h_blocks):
				y=int(h_block*h_block_len)
				x=int(w_block*w_block_len)
				#print(y)
				#print(x)				
				#deb.prints([y:y+self.patch_len])
				self.im_reconstructed[y:y+self.patch_len,x:x+self.patch_len]=patches_block[h_block,w_block,:,:]
				count+=1

		self.im_reconstructed_rgb=self.im_gray_idx_to_rgb(self.im_reconstructed)
		if self.debug>=3: 
			deb.prints(count)
			deb.prints(self.im_reconstructed_rgb.shape)

		cv2.imwrite('../results/reconstructed/im_reconstructed_rgb_'+subset+'_'+mode+self.report['exp_id']+'.png',self.im_reconstructed_rgb.astype(np.uint8))

	def im_gray_idx_to_rgb(self,im):
		out=np.zeros((im.shape+(3,)))
		for chan in range(0,3):
			for clss in range(0,self.class_n):
				out[:,:,chan][im==clss]=np.array(self.im_gray_idx_to_rgb_table[clss][1][chan])
		deb.prints(out.shape)
		out=cv2.cvtColor(out.astype(np.uint8),cv2.COLOR_RGB2BGR)
		return out


class NetModel(NetObject):
	def __init__(self, batch_size_train=32, batch_size_test=200, epochs=30, patience=30, eval_mode='metrics', *args, **kwargs):
		super().__init__(*args, **kwargs)
		if self.debug >= 1:
			print("Initializing Model instance")
		self.metrics = {'train': {}, 'test': {}}
		self.batch = {'train': {}, 'test': {}}
		self.batch['train']['size'] = batch_size_train
		self.batch['test']['size'] = batch_size_test
		self.eval_mode = eval_mode
		self.epochs = epochs
		self.early_stop={'best':0,
					'count':0,
					'signal':False,
					'patience':patience}

	def transition_down(self, pipe, filters):
		pipe = Conv2D(filters, (3, 3), strides=(2, 2), padding='same')(pipe)
		pipe = keras.layers.BatchNormalization(axis=3)(pipe)
		pipe = Activation('relu')(pipe)
		#pipe = Conv2D(filters, (1, 1), padding='same')(pipe)
		#pipe = keras.layers.BatchNormalization(axis=3)(pipe)
		#pipe = Activation('relu')(pipe)
		
		return pipe

	def dense_block(self, pipe, filters):
		pipe = Conv2D(filters, (3, 3), padding='same')(pipe)
		pipe = keras.layers.BatchNormalization(axis=3)(pipe)
		pipe = Activation('relu')(pipe)
		return pipe

	def transition_up(self, pipe, filters):
		pipe = Conv2DTranspose(filters, (3, 3), strides=(
			2, 2), padding='same')(pipe)
		pipe = keras.layers.BatchNormalization(axis=3)(pipe)
		pipe = Activation('relu')(pipe)
		#pipe = Dropout(0.2)(pipe)
		#pipe = Conv2D(filters, (1, 1), padding='same')(pipe)
		#pipe = keras.layers.BatchNormalization(axis=3)(pipe)
		#pipe = Activation('relu')(pipe)
		return pipe

	def concatenate_transition_up(self, pipe1, pipe2, filters):
		pipe = keras.layers.concatenate([pipe1, pipe2], axis=3)
		pipe = self.transition_up(pipe, filters)
		return pipe

	def build(self):
		in_im = Input(shape=(self.patch_len, self.patch_len, self.channel_n))
		filters = 64
#		filters = 32

		#pipe = {'fwd': [], 'bckwd': []}
		c = {'init_up': 0, 'up': 0}
		pipe=[]

		# ================== Transition Down ============================ #
		pipe.append(self.transition_down(in_im, filters))  # 0 16x16
		pipe.append(self.transition_down(pipe[-1], filters*2))  # 1 8x8
		pipe.append(self.transition_down(pipe[-1], filters*4))  # 2 4x4
		pipe.append(self.transition_down(pipe[-1], filters*8))  # 2 4x4
		
		
		c['down']=len(pipe)-1 # Last down-layer idx
		
		# =============== Dense block; no transition ================ #
		#pipe.append(self.dense_block(pipe[-1], filters*16))  # 3 4x4

		# =================== Transition Up ============================= #
		c['up']=c['down'] # First up-layer idx 
		
		pipe.append(self.concatenate_transition_up(pipe[-1], pipe[c['up']], filters*8))  # 4 8x8
		c['up']-=1
		pipe.append(self.concatenate_transition_up(pipe[-1], pipe[c['up']], filters*4))  # 4 8x8
		c['up']-=1
		pipe.append(self.concatenate_transition_up(pipe[-1], pipe[c['up']], filters*2))  # 5
		c['up']-=1
		pipe.append(self.concatenate_transition_up(pipe[-1], pipe[c['up']], filters))  # 6

		out = Conv2D(self.class_n, (1, 1), activation='softmax',
					 padding='same')(pipe[-1])

		self.graph = Model(in_im, out)
		print(self.graph.summary())

	def compile(self, optimizer, loss='binary_crossentropy', metrics=['accuracy',metrics.categorical_accuracy],loss_weights=None):
		loss_weighted=weighted_categorical_crossentropy(loss_weights)
		self.graph.compile(loss=loss_weighted, optimizer=optimizer, metrics=metrics)

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

	def early_stop_check(self,metrics,epoch,most_important='average_acc'):

		if metrics[most_important]>=self.early_stop['best']:
			self.early_stop['best']=metrics[most_important]
			self.early_stop['overall_acc']=metrics['overall_acc']
			self.early_stop['f1_score']=metrics['f1_score']
			self.early_stop['average_acc']=metrics['average_acc']
			
			self.early_stop['count']=0
			print("Best metric updated")
			data.metrics_write_to_txt(metrics,epoch)
			data.im_reconstruct(subset='test',mode='prediction')
		else:
			self.early_stop['count']+=1
			if self.early_stop["count"]>=self.early_stop["patience"]:
				self.early_stop["signal"]=True
			else:
				self.early_stop["signal"]=False
	# test_metrics_evaluate(self,data,metrics,epoch):
			
			
	def train_loop(self, data):
		print('Start the training')
		cback_tboard = keras.callbacks.TensorBoard(
			log_dir='../summaries/', histogram_freq=0, batch_size=self.batch['train']['size'], write_graph=True, write_grads=False, write_images=False)

		batch = {'train': {}, 'test': {}}
		self.batch['train']['n'] = data.patches['train']['in'].shape[0] // self.batch['train']['size']
		self.batch['test']['n'] = data.patches['test']['in'].shape[0] // self.batch['test']['size']

		data.patches['test']['n']=data.patches['test']['label'].shape[0]
		data.patches['train']['n']=data.patches['train']['label'].shape[0]


		data.patches['test']['prediction']=np.zeros_like(data.patches['test']['label'])
		count,unique=np.unique(data.patches['test']['label'].argmax(axis=3),return_counts=True)
		print("count,unique",count,unique)
		deb.prints(data.patches['test']['label'].shape)
		deb.prints(self.batch['test']['n'])
		
		data.im_reconstruct(subset='test',mode='label')
		#for epoch in [0,1]:
		for epoch in range(self.epochs):

			self.metrics['train']['loss'] = np.zeros((1, 2))
			self.metrics['test']['loss'] = np.zeros((1, 2))

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

				#if batch_id % 50 == 0:
				#	self.test_metrics_evaluate(data,metrics,epoch)
			# Average epoch loss
			self.metrics['train']['loss'] /= self.batch['train']['n']

			data.patches['test']['prediction']=np.zeros_like(data.patches['test']['label'])
			self.batch_test_stats=True

			for batch_id in range(0, self.batch['test']['n']+1):
				idx0 = batch_id*self.batch['test']['size']
				idx1 = (batch_id+1)*self.batch['test']['size']


				# This is for last batch
				#if idx0>=data.patches['test']['n']:
				#	break
				if idx1>data.patches['test']['n']:
					idx1=data.patches['test']['n']
					
				#deb.prints(data.patches['test']['label'].shape)
				#print(idx0,idx1)
				batch['test']['in'] = data.patches['test']['in'][idx0:idx1]
				batch['test']['label'] = data.patches['test']['label'][idx0:idx1]

				#deb.prints(batch['test']['label'].shape)
				if self.batch_test_stats:
					self.metrics['test']['loss'] += self.graph.test_on_batch(
						batch['test']['in'], batch['test']['label'])		# Accumulated epoch




				#batch['test']['prediction']=self.graph.predict(batch['test']['in'],batch_size=self.batch['test']['size'])
				data.patches['test']['prediction'][idx0:idx1]=self.graph.predict(batch['test']['in'],batch_size=self.batch['test']['size'])

				
				# if (batch_id % 4 == 0) and (epoch % 3 == 0):
				# 	print("Saving image, batch id={}, epoch={}".format(batch_id,epoch))
				# 	#print(data.patches['test']['prediction'][idx0].argmax(axis=2).astype(np.uint8)*50.shape)
				# 	cv2.imwrite("../results/pred"+str(batch_id)+".png",data.patches['test']['prediction'][idx0].argmax(axis=2).astype(np.uint8)*50)
				# 	cv2.imwrite("../results/label"+str(batch_id)+".png",data.patches['test']['label'][idx0].argmax(axis=2).astype(np.uint8)*50)
			
			deb.prints(data.patches['test']['label'].shape)		
			deb.prints(idx1)
			print("Epoch={}".format(epoch))	
			
		
			# Get test metrics
			metrics=data.metrics_get(data.patches['test'])
			
			# Check early stop and store results if they are the best
			self.early_stop_check(metrics,epoch)

			#self.test_metrics_evaluate(data.patches['test'],metrics,epoch)
			if self.early_stop['signal']==True:
				deb.prints(self.early_stop['overall_acc'])
				deb.prints(self.early_stop['average_acc'])
				deb.prints(self.early_stop['f1_score'])
				break
			print('oa={}, aa={}, f1={}'.format(metrics['overall_acc'],metrics['average_acc'],metrics['f1_score']))
		
			# Average epoch loss
			self.metrics['test']['loss'] /= self.batch['test']['n']
			print("Train loss={}, Test loss={}".format(self.metrics['train']['loss'],self.metrics['test']['loss']))

			


flag = {"data_create": True, "label_one_hot": True}
if __name__ == '__main__':
	#
	data = Dataset(patch_len=args.patch_len, patch_step_train=args.patch_step_train,
		patch_step_test=args.patch_step_test,exp_id=args.exp_id)
	if flag['data_create']:
		data.create()

	adam = Adam(lr=0.0001, beta_1=0.9)
	model = NetModel(epochs=args.epochs, patch_len=args.patch_len,
					 patch_step_train=args.patch_step_train, eval_mode=args.eval_mode,
					 batch_size_train=args.batch_size_train,batch_size_test=args.batch_size_test,
					 patience=args.patience)
	model.build()
	model.loss_weights=np.array([0.21159622, 0.13360889, 0.17312638, 0.29637921, 0.1852893])
	metrics=['accuracy']
	#metrics=['accuracy',fmeasure,categorical_accuracy]
	model.compile(loss='binary_crossentropy',
				  optimizer=adam, metrics=metrics,loss_weights=model.loss_weights)
	if args.debug:
		deb.prints(np.unique(data.patches['train']['label']))
		deb.prints(data.patches['train']['label'].shape)
	model.train(data)
