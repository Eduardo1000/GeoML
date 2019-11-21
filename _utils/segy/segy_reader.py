import os
import numpy as np
from obspy.io.segy.segy import _read_segy


class SegyReader():

    def load_img(self, img_path, label):
        """
        Reads and Normalize a seismogram from the given segy file.
        :param img_path: a path to the segy file.
        :param label: a label to the segy.
        :return: seismogram image as numpy array normalized between 0-1.
        """
        segy = _read_segy(img_path)
        _traces = list()
        for trace in segy.traces:
            _traces.append(trace.data)
        x = np.asarray(_traces, dtype=np.float32)
        std = x.std()
        x -= x.mean()
        x /= std
        x *= 0.1
        x += .5
        x = np.clip(x, 0, 1)
        x = np.expand_dims( x, axis = 0 )
        return x.T, label
