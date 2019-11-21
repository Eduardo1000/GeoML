import pandas as pd
from sklearn.metrics import precision_score, recall_score, classification_report, confusion_matrix, f1_score

class Metrics:
	def __init__(self, prediction_csv, target_csv):
		self.pred = pd.read_csv(prediction_csv, names = ['filename', 'class'])
		self.pred['filename'] = self.pred['filename'].apply(lambda x: x.split('/')[-1])
		self.target = pd.read_csv(target_csv, names = ['filename', 'class'] )

		self.classes = ['good','bad','ugly']

		df = self.pred.set_index('filename').join(self.target.set_index('filename'), rsuffix='_other' ).dropna()
		self.pred = df['class']
		self.target = df['class_other']

	def precision(self):
		return precision_score( self.target, self.pred, average=None, labels=self.classes )

	def recall(self) :
		return recall_score( self.target, self.pred, average=None, labels=self.classes )

	def f1(self, average='weighted'):
		return f1_score( self.target, self.pred, average=average, labels=self.classes )

	def confusion_matrix(self):
		return confusion_matrix( self.target, self.pred, labels=self.classes )

	def report(self) :
		print( self.confusion_matrix() )
		print( classification_report( self.target, self.pred ) )
