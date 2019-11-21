from invoke import task
import os
import json
import sys

from seismogram_classifiers._miniception.net import Model

sys.path.append('..')


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Main:
    def __init__(self):
        # Initial variables
        self.config_path = 'seismogram_classifiers/_config/'
        self.dataset_path = '../dataset/'
        self.result_path = ''

    def load_config(self, config):
        print('Loading parameters from %s%s.json' % (self.config_path, config))
        json_path = os.path.join(self.config_path, config) + '.json'
        with open(json_path) as f:
            JSON = json.load(f)
        return dotdict(JSON)


@task
def predict(context, data, save_file=None, checkpoint='miniception_D_1571316181.050129',
            gpu_config='gpu', params_config='miniception_R17', img=False):
    """
    Function to Predict Seismogram Class

    Receives a given data_folder and save the results in a given file.

    :param data: abs path to the desired data folder
    :param save_file: filename path to save the predictions(*.csv)
    :return: save the predictions in a given file
    """

    m = Main()

    if save_file is None:
        save_file = os.path.join(data, 'results.csv')

    gpu = m.load_config(gpu_config)
    params = m.load_config(params_config)

    model = Model(gpu, params, checkpoint)
    # Predict Labels
    filenames, preds = model.predict(data, segy=not img)
    # Save Predictions
    model.save_result(filenames, preds, save_file)


@task
def refine(context, data, labels, save_file=None, checkpoint='miniception_D_1571316181.050129',
           gpu_config='gpu', params_config='miniception_R17', img=False):
    """
    Function to Refine the Classifier.

    Receives the dataset path and a classification file, retrain the model over the images and labels presented
    in the classification file and returns the predictions for the whole dataset folder.

    :param data: path to the desired data folder
    :param labels:  path to the classification file.
        The classification file contains the information from the filename and class to each seismogram.
    :param save_file:  filename path to save the predictions
    :return: save the predictions in a given file
    """

    m = Main()

    if save_file is None:
        save_file = os.path.join(data, 'results.csv')

    gpu = m.load_config(gpu_config)
    params = m.load_config(params_config)
    params.learning_rate = 1e-6

    model = Model(gpu, params, checkpoint)
    # Retrain the Network
    model.train(data, labels, segy=not img)
    # Predict Labels
    filenames, dev_pred = model.predict(data, segy=not img)
    # Save Predictions
    model.save_result( filenames, dev_pred, save_file)


@task
def train(context, data, labels, checkpoint=None, gpu_config='gpu', params_config='miniception_R17', img=False):
    """
    Function to Train the Classifier.

    Receives the dataset path and a classification file, train the model over the labels presented
    in the classification file.

    :param data: path to the desired data folder
    :param labels:  path to the classification file.
        The classification file contains the informations from the filename and class to each sismogram.
    :return: save the predictions in a given file
    """

    m = Main()

    gpu = m.load_config(gpu_config)
    params = m.load_config(params_config)

    model = Model(gpu, params, checkpoint)
    # Train Model
    model.train(data, labels, segy=not img)
