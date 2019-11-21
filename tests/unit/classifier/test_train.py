from seismogram_classifiers._miniception.net import Model
from tasks.classifier import Main
import pandas as pd
import pytest
from invoke import run
import os


def test_train():
	data = '_utils/segy/data'
	labels_before = '_utils/segy/data/labels.csv'
	labels_after = '_utils/segy/data/results_train.csv'
	cmd = f"inv classifier.train -d {data} -l {labels_before}"
	run( cmd, hide = True, warn = True )

	dataset = pd.read_csv( labels_after, names = ['filename', 'class'] )
	dataset_2 = pd.read_csv( f'{data}/results.csv', names = ['filename', 'class'] )

	assert dataset['filename'].values.sort( ) == dataset_2['filename'].values.sort( )
	assert dataset['class'].values.sort( ) == dataset_2['class'].values.sort( )

@pytest.mark.parametrize(
	"data, label",[
		('_utils/segy/data', '_utils/segy/data/labels.csv'),
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', 'temp/results/test_png.txt -i'),
])
def test_save_train_to_correct_folder_train(data, label) :
	cmd = f"inv classifier.train -d {data} -l {label}"
	run( cmd, hide = True, warn = True )

	assert os.path.exists( f'seismogram_classifiers/_checkpoints/{data.split("/")[-1]}/model.meta' )

@pytest.mark.parametrize(
	"data, labels, checkpoint, gpu, params, image",[
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', 'temp/results/test_png.txt', '', '', '', '-i'),
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia/', 'temp/results/test_png.txt', '', '', '', '-i'),
		('_utils/segy/data', '_utils/segy/data/labels.csv', '', '', '', ''),
		('_utils/segy/data/', '_utils/segy/data/labels.csv', '', '', '', ''),
])
def test_predict_right_parameter(data, labels, checkpoint, gpu, params, image) :
	cmd = f"inv classifier.train -d {data} {labels} {checkpoint} {gpu} {params} {image}"
	result = run( cmd, hide = True, warn = True )
	assert result.ok

@pytest.mark.parametrize(
	"data, labels, checkpoint, gpu, params, image",[
		('_utils/segy/data', '', '', '', '', '-i'), ('_utils/segy/data', '_utils/segy/data/labels.csv', '', '', '', '-i'),
		('_utils/segy/WRONG', '', '', '', '', ''), ('', '_utils/segy/data/labels.csv', '', '', '', '-i'),
		('', '_utils/segy/data/labels.csv', '', '', '', ''),
		('_utils/segy/WRONG', '_utils/segy/data/labels.csv', '', '', '', ''),
		('_utils/segy/data', '_utils/segy/data/WRONG.csv', '', '', '', ''),

		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', '', '', '', '', ''),
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', 'temp/results/test_png.txt', '', '', '', ''),
		('', 'temp/results/test_png.txt', '', '', '', '-i'),
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', 'temp/results/WRONG.txt', '', '', '', '-i'),
		('temp/dataset/WRONG', 'temp/results/test_png.txt', '', '', '', '-i'),
		# inverse labels
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', '_utils/segy/data/labels.csv', '', '', '', '-i'),
		('_utils/segy/data', 'temp/results/test_png.txt', '', '', '', ''),
		# not_found segy arguments
		('_utils/segy/data', '_utils/segy/data/labels.csv', '-c checkpoint_not_found', '', '', ''),
		('_utils/segy/data', '_utils/segy/data/labels.csv', '', '-g gpu_not_found', '', ''),
		('_utils/segy/data', '_utils/segy/data/labels.csv', '', '', '-p miniception_not_found', ''),
		# Not indicated segy data_folder or labels
		('_utils/segy/data', '', '', '', '', ''),
		('_utils/segy/data', '', '-c miniception_D_1571316181.050129', '', '', ''),
		('_utils/segy/data', '', '', '-g gpu', '', ''),
		('_utils/segy/data', '', '', '', '-p miniception_R17', ''),
		('', '_utils/segy/data/labels.csv', '', '', '', ''),
		('', '_utils/segy/data/labels.csv', '-c miniception_D_1571316181.050129', '', '', ''),
		('', '_utils/segy/data/labels.csv', '', '-g gpu', '', ''),
		('', '_utils/segy/data/labels.csv', '', '', '-p miniception_R17', ''),
		# not_found image arguments
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', 'temp/results/test_png.txt', '-c checkpoint_not_found', '', '', '-i'),
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', 'temp/results/test_png.txt', '', '-g gpu_not_found', '', '-i'),
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', 'temp/results/test_png.txt', '', '', '-p miniception_not_found', '-i'),
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', 'temp/results/test_png.txt', '-c checkpoint_not_found', '', '', ''),
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', 'temp/results/test_png.txt', '', '-g gpu_not_found', '', ''),
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', 'temp/results/test_png.txt', '', '', '-p miniception_not_found', ''),
		# Not indicated image data_folder or labels
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', '', '', '', '', '-i'),
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', '', '-c miniception_D_1571316181.050129', '', '', '-i'),
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', '', '', '-g gpu', '', '-i'),
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', '', '', '', '-p miniception_R17', '-i'),
		('', 'temp/results/test_png.txt', '-c miniception_D_1571316181.050129', '', '', '-i'),
		('', 'temp/results/test_png.txt', '', '-g gpu', '', '-i'),
		('', 'temp/results/test_png.txt', '', '', '-p miniception_R17', '-i'),
])
def test_predict_wrong_parameter(data, labels, checkpoint, gpu, params, image) :
	cmd = f"inv classifier.train -d {data} {labels} {checkpoint} {gpu} {params} {image}"
	result = run( cmd, hide = True, warn = True )
	assert not result.ok
