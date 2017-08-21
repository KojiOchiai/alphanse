# coding: utf-8

import math
import numpy as np
import chainer
import chainer.functions as F
from .distribution import Distribution, ConditionalDistribution
from . import sampling


# functions for distributions

def gaussian_kl_standard(q):
    # Dkl(q, N(0, 1))
    assert isinstance(q, Gaussian)
    mu, ln_var = q.get_params()
    var = F.exp(ln_var)
    return F.sum(mu * mu + var - ln_var - 1) * 0.5


# distributions

class Gaussian(Distribution):
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

    def kl(self, p):
        """Calculate KL-divergence between given two gaussian.
        D_{KL}(P||Q)=\frac{1}{2}\Bigl[(\mu_1-\mu_2)^T\Sigma_2^{-1}(\mu_1-\mu_2)
        + tr\bigl\{\Sigma_2^{-1}\Sigma_1 \bigr\}
        + \log\frac{|\Sigma_2|}{|\Sigma_1|} - d \Bigr]
        """
        assert isinstance(p, Gaussian)
        mu1, ln_var1 = self.get_params()
        mu2, ln_var2 = p.get_params()
        assert isinstance(mu2, chainer.Variable)
        assert isinstance(ln_var2, chainer.Variable)
        assert mu1.data.size == mu2.data.size
        assert ln_var1.data.size == ln_var2.data.size

        d = mu1.size
        var1 = F.exp(ln_var1)
        var2 = F.exp(ln_var2)
        return (F.sum((mu1 - mu2) * (mu1 - mu2) / var2) + F.sum(var1 / var2)
                + F.sum(ln_var2) - F.sum(ln_var1) - d) * 0.5

    def entropy(self):
        batch, dim = self.ln_var.shape
        return (F.sum(self.ln_var)
                + np.log(2 * np.pi * np.e) * 0.5 * dim * batch)

    def sample(self, n_sample=None):
        N = n_sample or self.n_sample
        return sampling.gaussian(F.tile(self.mu, (N, 1)),
                                 F.tile(self.ln_var, (N, 1)))


class Bernoulli(Distribution):
    def __init__(self, mu_raw):
        super().__init__()
        self.mu = F.sigmoid(mu_raw)
        self.mu_raw = mu_raw  # hold mu_raw to reduce calculation error

    def get_params(self):
        return (self.mu, self.mu_raw)

    def log_likelihood(self, x, reduce='sum'):
        self.check_reduce(reduce)
        # loss = x*F.log(self.mu) + (1 - x)*F.log(1 - self.mu)
        loss = - F.softplus(self.mu_raw) + x * self.mu_raw
        return self.reduce(loss, reduce)

    def kl(self, p):
        assert isinstance(p, Bernoulli)
        return F.sum(self.mu * F.log(self.mu / p.mu)
                     + (1 - self.mu) * F.log((1 - self.mu) / (1 - p.mu)))

    def entropy(self):
        return -F.sum(self.mu * F.log(self.mu)
                      + (1 - self.mu) * F.log((1 - self.mu)))


class Categorical(Distribution):
    def __init__(self, p_raw):
        super().__init__()
        self.p = F.softmax(p_raw)

    def get_params(self):
        return (self.p)

    def log_likelihood(self, x, reduce='sum'):
        self.check_reduce(reduce)
        p = self.p + 1e-7  # prevent zero
        loss = x * F.log(p)
        return self.reduce(loss, reduce)

    def kl(self, p):
        assert isinstance(p, Categorical)
        assert len(self.p) == len(p.p)
        assert np.all(0 < self.p.data)
        assert np.all(0 < p.p.data)
        return F.sum(self.p * F.log(self.p / p.p))

    def entropy(self):
        assert np.all(0 < self.p.data)
        return -F.sum(self.p * F.log(self.p))


class Concat(Distribution):
    def __init__(self, dists):
        self.dists = dists

    def get_params(self):
        param_list = [p.get_params() for p in self.dists]
        return [item for sublist in param_list for item in sublist]

    def log_likelihood(self, x_list):
        return sum([self.dists[i].log_likelihood(x_list[i])
                    for i in range(len(x_list))])

    def sample(self, n_sample=None):
        return F.concat([dist.sample(n_sample) for dist in self.dists])


# conditional distribution

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
