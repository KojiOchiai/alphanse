import numpy as np

from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check
import chainer.functions as F


class Categorical(function.Function):
    def __init__(self, tau=1):
        self.tau = tau

    def check_type_forward(self, in_type):
        type_check.expect(in_type.size() == 1)

    def noise(self, xp, shape):
        u = xp.random.rand(shape)
        return -xp.log(-np.log(u))

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        p, = inputs
        self.noise = self.noise(xp, p.shape)
        sample = F.softmax((F.log(p) + self.noise) / self.tau)
        return utils.force_array(sample),

    def backward(self, grad_output):
