import pandas as pd
import pytest
from invoke import run
import os


def test_refine_known_result():
	data = '_utils/segy/data'
	labels_before = '_utils/segy/data/labels.csv'
	labels_after = '_utils/segy/data/results_refine.csv'
	cmd = f"inv classifier.refine -d {data} -l {labels_before}"
	run( cmd, hide = True, warn = True )

	dataset = pd.read_csv( labels_after, names = ['filename', 'class'] )
	dataset_2 = pd.read_csv( f'{data}/results.csv', names = ['filename', 'class'] )

	assert dataset['filename'].values.sort( ) == dataset_2['filename'].values.sort( )
	assert dataset['class'].values.sort( ) == dataset_2['class'].values.sort( )

@pytest.mark.parametrize(
	"data, label, save_file",[
		('_utils/segy/data', '_utils/segy/data/labels.csv', 'temp/results/correct_folder_refine.csv'),
		('temp/dataset/FIGURAS_ML_PUC_2019_Ia', 'temp/results/test_png.txt -i', 'temp/results/correct_folder_img_train.csv'),
])
def test_save_refine_to_correct_folder(data, label, save_file) :
	cmd = f"inv classifier.refine -d {data} -l {label} -s {save_file}"
	run( cmd, hide = False, warn = True )

	assert os.path.exists( save_file )
	os.remove( save_file )
	assert os.path.exists( f'seismogram_classifiers/_checkpoints/{data.split( "/" )[-1]}/model.meta' )
