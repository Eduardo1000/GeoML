from seismogram_classifiers.metrics import Metrics
import pytest

def test_identical():
    m = Metrics('temp/results/train_png.txt', 'temp/results/train_png.txt')
    assert m.f1()==1

def test_opposite():
    m = Metrics('temp/results/opposite_0.txt', 'temp/results/opposite_1.txt')
    assert m.f1()==0

def test_f1():
    m = Metrics('temp/results/save_file.csv', 'temp/results/train_png.txt')
    assert m.f1()>0.95

def test_wrong_format():
    m = Metrics('Dockerfile', 'Dockerfile')
    with pytest.raises(ValueError):
        m.report()


