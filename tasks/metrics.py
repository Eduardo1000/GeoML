from invoke import task
from seismogram_classifiers.metrics import Metrics

@task(default=True)
def report(context, prediction, target):
	m = Metrics(prediction, target)
	m.report()

@task
def f1(context, prediction, target):
	m = Metrics(prediction, target)
	f1 = m.f1()
	print('F1 = ', f1  )
	return f1

@task
def recall(context, prediction, target):
	m = Metrics(prediction, target)
	f1 = m.f1()
	print('F1 = ', f1  )
	return f1


