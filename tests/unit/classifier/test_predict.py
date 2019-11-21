import os
import pandas as pd
import pytest
from invoke import run


def test_predict_known_result() :
	data = '_utils/segy/data'
	labels = '_utils/segy/data/results_predict.csv'
	cmd = f"inv classifier.predict -d {data}"
	run( cmd, hide = True, warn = True )

	dataset = pd.read_csv( labels, names = ['filename', 'class'] )
	dataset_2 = pd.read_csv( f'{data}/results.csv', names = ['filename', 'class'] )

	assert dataset['filename'].values.sort( ) == dataset_2['filename'].values.sort( )
	assert dataset['class'].values.sort( ) == dataset_2['class'].values.sort( )

@pytest.mark.parametrize(
	"data, save_file",[
		('_utils/segy/data', 'temp/results/correct_folder.csv'),
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia -i', 'temp/results/correct_folder_img.csv'),
])
def test_save_prediction_to_correct_folder(data, save_file) :
	cmd = f"inv classifier.predict -d {data} -s {save_file}"
	run( cmd, hide = True, warn = True )

	assert os.path.exists( save_file )
	os.remove( save_file )

@pytest.mark.parametrize(
	"data, checkpoint, gpu, params, image",[
		('_utils/segy/data', '', '', '', ''),
		('_utils/segy/data/', '', '', '', ''),
		('_utils/segy/data', '-c miniception_D_1571316181.050129', '', '', ''),
		('_utils/segy/data', '', '-g gpu', '', ''),
		('_utils/segy/data', '', '', '-p miniception_R17', ''),
		('_utils/segy/data', '', '-g gpu', '-p miniception_R17', ''),
		('_utils/segy/data', '-c miniception_D_1571316181.050129', '', '-p miniception_R17', ''),
		('_utils/segy/data', '-c miniception_D_1571316181.050129', '-g gpu', '-p miniception_R17', ''),
		('_utils/segy/data', '-c miniception_D_1571316181.050129', '', '-p miniception_R17', ''),

		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', '', '', '', '-i'),
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia/', '', '', '', '-i'),
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', '', '', '', '--img'),
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', '-c miniception_D_1571316181.050129', '', '', '--img'),
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', '', '-g gpu', '', '--img'),
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', '', '', '-p miniception_R17', '--img'),
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', '', '-g gpu', '-p miniception_R17', '--img'),
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', '-c miniception_D_1571316181.050129', '', '-p miniception_R17', '--img'),
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', '-c miniception_D_1571316181.050129', '-g gpu', '-p miniception_R17', '--img'),
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', '-c miniception_D_1571316181.050129', '', '-p miniception_R17', '--img'),
])
def test_predict_right_parameter(data, checkpoint, gpu, params, image) :
	cmd = f"inv classifier.predict -d {data} {checkpoint} {gpu} {params} {image}"
	result = run( cmd, hide = True, warn = True )
	assert result.ok


@pytest.mark.parametrize(
	"data, checkpoint, gpu, params, image",[
		('_utils/segy/data', '', '', '', '-i'), ('temp/dataset/FIGURAS_ML_PUC_2019_Ia', '', '', '', ''),
	    ('_utils/segy/ds', '', '', '', ''), ('temp/dataset/INCORRECT_PATH', '', '', '', '-i'),
		# not_found segy arguments
		('_utils/segy/data', '-c checkpoint_not_found', '', '', ''),
		('_utils/segy/data', '', '-g gpu_not_found', '', ''),
		('_utils/segy/data', '', '', '-p miniception_not_found', ''),
		('_utils/segy/data', '-c checkpoint_not_found', '', '', '-i'),
		('_utils/segy/data', '', '-g gpu_not_found', '', '-i'),
		('_utils/segy/data', '', '', '-p miniception_not_found', '-i'),
		# Not indicated segy data_folder
		('', '', '', '', ''),
		('', '-c miniception_D_1571316181.050129', '', '', ''),
		('', '', '-g gpu', '', ''),
		('', '', '', '-p miniception_R17', ''),
		('', '-c miniception_D_1571316181.050129', '', '', ''),
		('', '', '-g gpu', '', ''),
		('', '', '', '-p miniception_R17', ''),
		('', '-c miniception_D_1571316181.050129', '', '-p miniception_R17', ''),
		('', '', '-g gpu', '-p miniception_R17', ''),
		('', '-c miniception_D_1571316181.050129', '-g gpu', '-p miniception_R17', ''),
		# not_found image arguments
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', '-c checkpoint_not_found', '', '', '-i'),
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', '', '-g gpu_not_found', '', '-i'),
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', '', '', '-p miniception_not_found', '-i'),
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', '-c checkpoint_not_found', '', '', ''),
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', '', '-g gpu_not_found', '', ''),
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', '', '', '-p miniception_not_found', ''),
		# Not indicated image data_folder
		('', '', '', '', '-i'),
		('', '-c miniception_D_1571316181.050129', '', '', '-i'),
		('', '', '-g gpu', '', '-i'),
		('', '', '', '-p miniception_R17', '-i'),
		('', '-c miniception_D_1571316181.050129', '', '', '-i'),
		('', '', '-g gpu', '', '-i'),
		('', '', '', '-p miniception_R17', '-i'),
		('', '-c miniception_D_1571316181.050129', '', '-p miniception_R17', '-i'),
		('', '', '-g gpu', '-p miniception_R17', '-i'),
		('', '-c miniception_D_1571316181.050129', '-g gpu', '-p miniception_R17', '-i'),
])
def test_predict_wrong_parameter(data, checkpoint, gpu, params, image) :
	cmd = f"inv classifier.predict -d {data} {checkpoint} {gpu} {params} {image}"
	result = run( cmd, hide = True, warn = True )
	assert not result.ok
