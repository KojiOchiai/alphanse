# coding: utf-8

import numpy as np
import chainer


def variable(dim, value=0, batch=1):
    '''return chainre.Variable
    '''
    data = np.ones([batch, dim]).astype(np.float32)
    if 2 <= int(chainer.__version__.split('.')[0]):
        v = chainer.Variable(value * data)
    else:
        v = chainer.Variable(value * data, volatile='auto')
    return v


def parameter(dim, value=0, batch=1):
    '''return chainre.Parameter
    '''
    data = np.ones([batch, dim]).astype(np.float32)
    if 2 <= int(chainer.__version__.split('.')[0]):
        p = chainer.Parameter(value * data)
    else:
        raise NotImplementedError()
    return p


def one_hot(y, dim=10):
    y_onehot = np.zeros((y.size, dim), dtype=np.float32)
    y_onehot[np.arange(y.size), y] = 1
    return y_onehot
