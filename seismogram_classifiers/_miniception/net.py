import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

import seismogram_classifiers._miniception.miniception as model
from _utils.DatasetBuilder import DatasetBuilder
from seismogram_classifiers import _Base


class Model(_Base):
	def __init__(self, config, params, checkpoint=None):
		super().__init__(config)
		# Inicia Parâmetros Básicos
		self.params = params
		self.checkpoint = checkpoint
		self.sess = self.create_session()

		# Cria o Model
		self._build_model()

		# Define Variáveis
		self.saver = tf.train.Saver( )
		self.sess.run( tf.global_variables_initializer( ) )
		self.features_pl = tf.placeholder( tf.string, shape = (None,) )
		self.labels_pl = tf.placeholder( tf.int64, shape = (None,) )
		self.batch_size = 1

	def predict(self, x, segy=True, **kwargs):

		print( 'Restoring Checkpoint ', f'{self.checkpoint_path}{self.checkpoint}' )
		checkpoint = tf.train.latest_checkpoint( f'{self.checkpoint_path}{self.checkpoint}/' )
		self.saver.restore( self.sess, checkpoint )

		DB = DatasetBuilder( x, None, segy=segy )
		DB.load_from_dir()
		dataset = DB.make_dataset( self.features_pl, self.labels_pl, self.batch_size, training = False, segy=segy )
		init_op, next = DB.make_iterator( dataset )

		if len(DB._paths)==0 or len(DB._labels)==0:
			raise Exception("No data found. Check the data path. "
			                "If working with images, don't forget to pass -i argument")

		self.sess.run( init_op.initializer,
		               feed_dict = { self.features_pl : DB._paths,
		                             self.labels_pl   : DB._labels } )

		dev_pred = []

		testbar = tqdm( range( 0, len( DB._paths ), self.batch_size ) )
		for _ in testbar :
			testbar.set_description( "\rPredicting" )
			try :
				x_batch, y_batch = self.sess.run( next )

				x_batch = np.array( x_batch )

				feed_dict_batch = { self.x : x_batch, self.y : y_batch }
				valid_pred, logit = self.sess.run( [self.pred, self.output_logits], feed_dict = feed_dict_batch )

				dev_pred.extend( valid_pred )
			except Exception as e :
				print( e )
				dev_pred.extend( [0] )

		testbar.close( )

		return DB._paths, dev_pred

	def train(self, data, labels, segy=True):
		
		lr = self.params.learning_rate
		data_folder = data.split('/')[-1]
		if len(data_folder) == 0:
			data_folder = data.split('/')[-2]

		checkpoint_file = self.checkpoint_path + data_folder + '/model'

		if self.checkpoint:
			n_epochs = 1
			print('Restoring Checkpoint ', f'{self.checkpoint_path}{self.checkpoint}')
			checkpoint = tf.train.latest_checkpoint( f'{self.checkpoint_path}{self.checkpoint}' )
			self.saver.restore( self.sess, checkpoint )
		else:
			n_epochs = 40

		DB = DatasetBuilder( data, labels, segy=segy )
		dataset = DB.make_dataset( self.features_pl, self.labels_pl, self.batch_size, training = True, segy=segy,
		                           num_epochs = self.params.num_epochs )
		init_op, next = DB.make_iterator( dataset )

		if len(DB._paths)==0 or len(DB._labels)==0:
			raise Exception("No data found. Check the data path. "
			                "If working with images, don't forget to pass -i argument")

		self.sess.run( init_op.initializer,
		               feed_dict = { self.features_pl : DB._paths,
		                             self.labels_pl   : DB._labels } )

		pbar = tqdm(range(1, n_epochs + 1))
		for epoch in pbar :
			# Print the status message.
			pbar.set_description( f"\r- Training epoch: {format(epoch)} " )
			lr *= 0.999
			# Treinamento
			trainbar = tqdm( range(0, len(DB._paths), self.batch_size))
			for _ in trainbar:
				trainbar.set_description("\r- Training")
				try :
					x_batch, y_batch = self.sess.run( next )

					x_batch = np.array( x_batch )

					feed_dict_batch = { self.x : x_batch, self.y : y_batch, self.lr_pl : lr }
					self.sess.run( self.optimizer, feed_dict = feed_dict_batch )
				except Exception as e :
					print( e )

			trainbar.close( )

			if epoch == n_epochs :
				print( '\n\n Save last model \n' )
				print( 'Chackpoint saved in ', checkpoint_file)
				self.saver.save( self.sess, checkpoint_file )
		pbar.close( )

		self.checkpoint = data_folder

	def _build_model(self):
		# get model from input placeholder variable and number of classes
		self.x, self.y, self.lr_pl, self.output_logits, self.pred, self.model_description, self.tscores = model.make_model(
			n_classes=3, alpha=self.params.alpha, width=None, height=None, seed=self.seed,
			n_blocks=self.params.n_blocks, n_channels=self.params.n_channels
		)

		self.loss, self.accuracy, self.optimizer = model.make_model_loss(self.y, self.lr_pl, self.tscores)
		# define model name
		self.model_name = model.model_name + '_' + str(time.time())

	@staticmethod
	def save_result(filenames, predictions, save_file):
		df = pd.DataFrame([filenames, predictions]).T
		df.columns = ['file', 'prediction']
		class_invert = ['good','bad','ugly']
		for i in range(3):
			df['prediction'].replace(i, class_invert[i], inplace=True )

		print('\nSaving results on:', save_file)

		df.to_csv(save_file, index=False, header=None)

