import os
import tensorflow as tf
import numpy as np
from _utils.segy.segy_reader import SegyReader


class DatasetBuilder(object):

	def __init__(self, path, labels_path=None, batch_size=1, segy=True, max_size=-1):
		"""
		:param path:
		:param labels_path:
		:param batch_size:
		:param file_extension:
		:param max_size:
		"""
		if segy:
			self._file_extension = ('.sgy')
		else:
			self._file_extension = ('.jpg', '.jpeg', '.png')
		self.path = path
		self.labels_path = labels_path
		self._paths = list()
		self._labels = list()
		self._max_size = max_size
		self._batch_size = batch_size
		self.seed = 12345
		self.segy_reader = SegyReader()
		self.load_data()

	def load_data(self):
		if self.labels_path is None:
			self._labels = [0 for _ in range(len(self._paths))]
		else:
			with open(self.labels_path, 'r') as f:
				for row in f:
					name, label = row.split(",")
					label = label.lower().strip()
					# name = name.strip().replace(".png", ".segy")
					self._paths.append(os.path.join(self.path, name))
					if (label == "good") or (label == 0):
						self._labels.append(0)
					elif (label == "bad") or (label == 1):
						self._labels.append(1)
					elif (label == "ugly") or (label == 2):
						self._labels.append(2)
					else:
						raise ValueError("Label not recognized. Found in data: '{}'".format(label))
					if 0 < self._max_size == len(self._paths):
						break
		self._paths = np.asarray(self._paths)
		self._labels = np.asarray(self._labels)

	def load_from_dir(self) :
		self._paths = list( )
		self._labels = list( )
		for root, _, files in os.walk( self.path ) :
			for file in files :
				if not file.endswith( self._file_extension ) :
					continue
				file_path = os.path.join( root, file )
				self._paths.append( file_path )
				self._labels.append( 0 )

				if 0 < self._max_size == len( self._paths ) :
					break
		self._paths = np.asarray( self._paths )
		self._labels = np.asarray( self._labels )

	def process_path(self, filename, label) :
		img = tf.read_file( filename, name = 'read_file' )
		img = tf.image.decode_image( img, name = 'decode', channels = 3, expand_animations = False )
		img = tf.image.rgb_to_grayscale( img )
		img = tf.cast( 2 * (img / 255 - 0.5), tf.float32 )
		label = tf.cast( label, tf.int64 )
		return img, label

	def make_dataset(self, filename, labels, batch_size, training=True, segy=True, num_epochs=40) :
		dataset = tf.data.Dataset.from_tensor_slices( (filename, labels) )
		if training :
			dataset = dataset.shuffle( 110000, seed = self.seed )
		if segy :
			dataset = dataset.map( lambda filename, label :
								   tuple( tf.py_func( self.segy_reader.load_img, [filename, label], [tf.float32, tf.int64] ) ),
								   num_parallel_calls = 4 )
		else :
			dataset = dataset.map( self.process_path, num_parallel_calls = 4 )
		dataset = dataset.batch( batch_size )
		if training :
			dataset = dataset.repeat( num_epochs )
		dataset = dataset.prefetch( 10 )
		return dataset

	def make_iterator(self, training_dataset) :
		dataset_iterator = training_dataset.make_initializable_iterator( )
		next_element = dataset_iterator.get_next( )
		return dataset_iterator, next_element
