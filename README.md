# GeoML

Repository to Seismogram Noise Classification developed in parternership between PUC-Rio (Grupo Learn) and Petrobrás.

It's intended to be used with seismograms with the following extensions: '.sgy' , '.png' , '.jpg'

The image bellow show examples from three available classes to two different aquisitions. 

| Aquisition | Good | Bad | Ugly |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
| A |![Example Image A 0](_utils/example/sismoMod_Exp_00000008.png?raw=true "Image A 0") | ![Example Image A 1](_utils/example/sismoMod_Exp_00001976.png?raw=true "Image A 1") | ![Example Image A 2](_utils/example/sismoMod_Exp_00003978.png?raw=true "Image A 2") |
| B |![Example Image B 0](_utils/example/sismoMod_Exp_0002.png?raw=true "Image B 0")  |  ![Example Image B 1](_utils/example/sismoMod_Exp_0020.png?raw=true "Image B 1") | ![Example Image B 2](_utils/example/sismoMod_Exp_0990.png?raw=true "Image B 2") |

The model employeed is composed by 4 Miniception blocks, manually designed by specialists. The results report 93,00% F1-score for a Cross-Validation without data augmentation in Aquisition A and 83.7% F1-score with data augmentation in a Holdout in Aquisition B.

##
### 1 Setup Configuration
 
Build the Project Docker Image

This project is builded on a 1.14.0 tensorflow image. 
If your GPU is not compatible with 1.14, replace this version with the desired one.  
```
sh.build
```

Run the Project Docker Image

```
sh.run
```

##
### 2 Predict
```
inv classifier.predict -d 'data_dir' -s 'save_path'
```
data_dir = the relative path to the directory that contains the dataset.

save_path (optional argument) = the relative path save the predictions. If not informed, 
    the system will save the predictions within the 'data_dir' as name 'results.csv'

#### 3 Metrics
```
inv metrics.report -p 'prediction_file' -t 'target_file'
```

prediction_file = the relative path to the file that contains the predictions.

target_file = the relative path to the file that contains the target labels.

return: display the confusion matrix and the classification report, 
that contais f1-score, recall and precision for each class.

##
### 4 Labeling Tool
The learning methods are proposed to fit data to the Neural Network.

There are two Learning Methods available: Refine and Train, that are detailed in each subsection. 

Both methods requires labelling the data, that is the act of define the class of each seismogram. 
To perform labelling, the notebook "classifier" is good tool for that. 
The images bellow presents the process of labelling.

You must select a Dataset Path. In this path should be stored all the seismic data that you want to work.
You can also select the number of maximun Examples, that to refine is suggested to be 100, 
    but you can set it to 0 to work with the whole dataset.
![Example Image A 0](_utils/example/choose_folder.png?raw=true "Image A 0")

Next step is the process of manually labelling the seismic data by clicking in the three class buttons: 'good','bad',ugly'.

![Example Image A 0](_utils/example/labelling.png?raw=true "Image A 0")

When you are finished, click on the save button that will appear.

#### 5 Refine
Refine Method is proposed to be used in new aquisitions.
In its core, the Refine Method make use of a pretrained network and
    refine its weights to be especially designed to a new aquisition. 

The trained checkpoint will be stored on the folder 'seismogram_classifiers/_checkpoints/{data_dir}', 
where data_dir is the argument directory given in the command line.
If the checkpoint already exists, it will be overwritten.

```
inv classifier.refine -d 'data_dir' -l 'label_path' -s 'save_path'
```
data_dir = the relative path to the directory that contains the dataset.

label_path = the relative path to the filename with respective classes.
    If you don't have one, use the notebook "classifier".

save_path (optinal argument) = the relative path save the predictions. If not informed, 
    the system will save the predictions within the 'data_dir' as name 'results.csv'

#### 6 Train
Train Method is designed to construct the network from scratch. 
This function builds the architecture and initialize the weigths randomly, 
then learn from a given directory with respective labels.

The trained checkpoint will be stored on the folder 'seismogram_classifiers/_checkpoints/{data_dir}', 
where data_dir is the argument directory given in the command line.
If the checkpoint already exists, it will be overwritten.

```
inv classifier.train -d 'data_dir' -l 'label_path'
```
data_dir = the relative path to the directory that contains the dataset.

label_path = the relative path to the filename with respective classes.
    If you don't have one, use the notebook "classifier".
