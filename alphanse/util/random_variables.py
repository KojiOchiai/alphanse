# coding: utf-8

import time
import math
import copy
import numpy as np
import chainer
import chainer.functions as F

# utility functions
def constant_variable(dim, value=0, batch=1):
    '''return chainre.Variable
    '''
    data = np.ones([batch, dim]).astype(np.float32)
    if 2 <= int(chainer.__version__.split('.')[0]):
        v = chainer.Variable(value * data)
    else:
        v = chainer.Variable(value * data, volatile='auto')
    return v

def set_sample_number(distribution, N):
    dist = copy.deepcopy(distribution)
    dist.n_sample = N
    return dist
    
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
    mu1, ln_var1 = q.get_param()
    mu2, ln_var2 = p.get_param()
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

# distribution
class Distribution(object):
    def __init__(self):
        raise NotImplementedError()

    def get_param(self):
        raise NotImplementedError()
    
    def likelihood(self, x, reduce='sum'):
        return F.exp(self.log_likelihood(x, reduce=reduce))

    def log_likelihood(self, x, reduce='sum'):
        raise NotImplementedError()

    def nll(self, x, reduce='sum'):
        return -self.log_likelihood(x, reduce=reduce)

    def check_reduce(self, reduce):
        if reduce not in ('sum', 'no'):
            raise ValueError(
                "only 'sum' and 'no' are valid for 'reduce', but '%s' is "
                'given' % reduce)
        else:
            return True

    def reduce(self, loss, reduce):
        if reduce == 'sum':
            return F.sum(loss)
        else:
            return loss

    def to_gpu(self, gpu):
        for param in self.get_param():
            param.to_gpu(gpu)
            
    def to_cpu(self):
        for param in self.get_param():
            param.to_cpu()

class ContinuousDistribution(Distribution):
    def __init__(self):
        super().__init__()
    
class DiscreteDistribution(Distribution):
    def __init__(self):
        super().__init__()

class Gaussian(ContinuousDistribution):
    def __init__(self, mu, ln_var, n_sample=1, clip=False,
                 clip_min=0.01, clip_max=10):
        if clip:
            ln_clip_min = math.log(clip_min)
            ln_clip_max = math.log(clip_max)
            ln_var = F.clip(ln_var, ln_clip_min, ln_clip_max)
        self.mu = mu
        self.ln_var = ln_var
        self.n_sample = n_sample
        
    def get_param(self):
        return (self.mu, self.ln_var)
    
    def log_likelihood(self, x, reduce='sum'):
        self.check_reduce(reduce)
        x_prec = F.exp(-self.ln_var)
        x_diff = x - self.mu
        x_power = (x_diff * x_diff) * x_prec * -0.5
        loss = - (self.ln_var + math.log(2 * math.pi)) / 2 + x_power
        return self.reduce(loss, reduce)
    
    def sample(self, n_sample=None):
        N = n_sample or self.n_sample
        return F.gaussian(F.tile(self.mu, (N, 1)),
                          F.tile(self.ln_var, (N, 1)))

class Bernoulli(DiscreteDistribution):
    def __init__(self, mu_raw):
        self.mu = F.sigmoid(mu_raw)
        self.mu_raw = mu_raw # hold mu_raw to reduce calculation error

    def get_param(self):
        return (self.mu, self.mu_raw)
    
    def log_likelihood(self, x, reduce='sum'):
        self.check_reduce(reduce)
        # loss = x*F.log(self.mu) + (1 - x)*F.log(1 - self.mu)
        loss = - F.softplus(self.mu_raw) + x * self.mu_raw
        return self.reduce(loss, reduce)
        
class Categorical(DiscreteDistribution):
    def __init__(self, p_raw):
        self.p = F.softmax(p_raw)

    def get_param(self):
        return (self.p)
    
    def log_likelihood(self, x, reduce='sum'):
        self.check_reduce(reduce)        
        p = self.p + 1e-7 # prevent zero
        loss = x*F.log(p)
        return self.reduce(loss, reduce)

# conditional distribution
class ConditionalDistribution(chainer.Chain):
    def __init__(self, model):
        super().__init__(
            model=model,
        )

    def __call__(self, x):
        raise NotImplementedError()

class ConditionalGaussian(ConditionalDistribution):
    def __init__(self, model, clip=False, clip_min=0.01, clip_max=10):
        super().__init__(model=model)
        self.clip = clip
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self, x):
        mu, ln_var = self.model(x)
        return Gaussian(mu, ln_var, clip=self.clip,
                        clip_min=self.clip_min, clip_max=self.clip_max)

class ConditionalBernoulli(ConditionalDistribution):
    def __call__(self, x):
        mu_raw = self.model(x)
        return Bernoulli(mu_raw)

class ConditionalCategorical(ConditionalDistribution):
    def __call__(self, x):
        p = self.model(x)
        return Categorical(p)
