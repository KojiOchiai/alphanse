# coding: utf-8

import math
import numpy as np
import copy
import chainer
import chainer.functions as F


def constant_variable(dim, value=0, batch=1):
    '''return chainre.Variable
    '''
    data = np.ones([batch, dim]).astype(np.float32)
    if 2 <= int(chainer.__version__.split('.')[0]):
        v = chainer.Variable(value * data)
    else:
        v = chainer.Variable(value * data, volatile='auto')
    return v

def constant_parameter(dim, value=0, batch=1):
    '''return chainre.Parameter
    '''
    data = np.ones([batch, dim]).astype(np.float32)
    if 2 <= int(chainer.__version__.split('.')[0]):
        p = chainer.Parameter(value * data)
    else:
        raise NotImplementedError()
    return p

def parameter(dim, value=0, batch=1):
    '''return chainre.Parameter
    '''
    data = np.ones([batch, dim]).astype(np.float32)
    v = chainer.Parameter(value * data)
    return v

def set_sample_number(distribution, N):
    distribution.n_sample = N
    return distribution
    
def one_hot(y, size=10):
    y_onehot = np.zeros((y.size, size), dtype=np.float32)
    y_onehot[np.arange(y.size), y] = 1
    return y_onehot

def gaussian_kl_standard(q):
    return F.gaussian_kl_divergence(q.mu, q.ln_var)

def gaussian_kl(q, p):
    """Calculate KL-divergence between given two gaussian.
    D_{KL}(P||Q)=\frac{1}{2}\Bigl[(\mu_1-\mu_2)^T\Sigma_2^{-1}(\mu_1-\mu_2) + tr\bigl\{\Sigma_2^{-1}\Sigma_1 \bigr\} 
    + \log\frac{|\Sigma_2|}{|\Sigma_1|} - d \Bigr]
    """
    assert isinstance(q, Gaussian)
    assert isinstance(p, Gaussian)
    mu1, ln_var1 = q.get_params()
    mu2, ln_var2 = p.get_params()
    assert isinstance(mu1, chainer.Variable)
    assert isinstance(ln_var1, chainer.Variable)
    assert isinstance(mu2, chainer.Variable)
    assert isinstance(ln_var2, chainer.Variable)
    assert mu1.data.size == mu2.data.size
    assert ln_var1.data.size == ln_var2.data.size
    assert mu1.data.size == ln_var1.data.size
    
    d = mu1.size
    var1 = F.exp(ln_var1)
    var2 = F.exp(ln_var2)
    return (F.sum((mu1-mu2)*(mu1-mu2)/var2) + F.sum(var1/var2)
            + F.sum(ln_var2) - F.sum(ln_var1) - d) * 0.5
