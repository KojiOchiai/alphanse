# coding: utf-8

import math
import numpy as np
import copy
import chainer
import chainer.functions as F


class Distribution(chainer.Link):
    def __init__(self):
        super().__init__()

    def __setattr__(self, name, value):
        if isinstance(value, chainer.Parameter):
            value.name = name
            if not self._cpu:
                value.to_gpu(self._device_id)
            self._params.add(name)
            self._persistent.discard(name)
        super().__setattr__(name, value)

    def __delattr__(self, name):
        self._params.discard(name)
        self._persistent.discard(name)
        super().__delattr__(name)
         
    def get_params(self):
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
                      
    def to_gpu(self, device=None):
        for param in self.get_params():
            param.to_gpu(device)
            
    def to_cpu(self):
        for param in self.get_params():
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
        super().__init__()
        if clip:
            ln_clip_min = math.log(clip_min)
            ln_clip_max = math.log(clip_max)
            ln_var = F.clip(ln_var, ln_clip_min, ln_clip_max)
        self.mu = mu
        self.ln_var = ln_var
        self.n_sample = n_sample
        
    def get_params(self):
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
        super().__init__()
        self.mu = F.sigmoid(mu_raw)
        self.mu_raw = mu_raw # hold mu_raw to reduce calculation error

    def get_params(self):
        return (self.mu, self.mu_raw)
    
    def log_likelihood(self, x, reduce='sum'):
        self.check_reduce(reduce)
        # loss = x*F.log(self.mu) + (1 - x)*F.log(1 - self.mu)
        loss = - F.softplus(self.mu_raw) + x * self.mu_raw
        return self.reduce(loss, reduce)
        
class Categorical(DiscreteDistribution):
    def __init__(self, p_raw):
        super().__init__()
        self.p = F.softmax(p_raw)

    def get_params(self):
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
