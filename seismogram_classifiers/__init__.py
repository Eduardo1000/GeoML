import os
import tensorflow as tf
from keras import backend as K
import numpy as np


class _Base:
    def __init__(self, config, **kwargs):
            self.config = config

            # Define Global Variables
            self.checkpoint_path = 'seismogram_classifiers/_checkpoints/'

            os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

            tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)

            self.seed = 12345
            np.random.seed(self.seed)

    def create_session(self):
        # Set GPU device, Limit GPU's Memory and Growth of GPU's memory
        default_config = tf.ConfigProto(log_device_placement=self.config.log_device_placement)
        default_config.gpu_options.visible_device_list = self.config.gpu

        # default_config.gpu_options.per_process_gpu_memory_fraction = self.config.gpu_memory
        default_config.gpu_options.allow_growth = self.config.allow_growth
        sess = tf.Session(config=default_config)

        # Bind Keras to Tensorflow Configurations
        K.set_session(sess)
        return sess

    def predict(self, x, **kwargs):
        print('not implemented')
        raise NotImplementedError

    def refine(self, x, y):
        raise NotImplementedError
