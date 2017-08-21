# coding: utf-8

import chainer
import chainer.functions as F


# functions for distributions

def set_sample_number(distribution, N):
    distribution.n_sample = N
    return distribution


def Dkl(q, p):
    return q.kl(p)


def expectation(density, func=(lambda x: x), sample=1):
    expected_value = 0
    for l in range(sample):
        z = density.sample()
        expected_value += func(z)
    return expected_value / sample


def entropy(distribution, sample=1):
    assert isinstance(distribution, Distribution)
    if hasattr(distribution, 'entropy'):
        return distribution.entropy()
    return - expectation(distribution,
                         distribution.log_likelihood, sample)


# distributions

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

    def kl(self, p):
        raise NotImplementedError()

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


# conditional distribution

class ConditionalDistribution(chainer.Chain):
    def __init__(self, model):
        super().__init__(
            model=model,
        )

    def __call__(self, x):
        raise NotImplementedError()


# likelihood

class LogLikelihood:
    def __init__(self, distribution, observed):
        assert isinstance(distribution, ConditionalDistribution)
        self.distribution = distribution
        self.observed = observed

    def __call__(self, condition):
        return self.distribution(condition).log_likelihood(self.observed)
