# coding: utf-8

import numpy as np
from scipy.misc import logsumexp
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
import varit.random_variables as rv


class MVAE(chainer.Chain):
    def __init__(self, qzgxy, pxgz, pygz):
        super().__init__(
            qzgxy=qzgxy,
            pxgz=pxgz,
            pygz=pygz,
        )
    
    def reconstract_multiple(self, x, y, sample=False):
        pz = self.qzgxy(F.concat([x, y]))
        return self.generate(pz, sample)
                        
    def generate(self, pz, sample=False):
        if sample:
            z = pz.sample()
        else:
            z = pz.mu
        px = self.pxgz(z)
        py = self.pygz(z)
        return px, py

    def lower_bound(self, x, y, C=1.0, sample=1):
        # loss function
        qz = self.qzgxy(F.concat([x, y]))
        return self.free_energy(qz, x, y, C=C, sample=sample)

    def free_energy(self, qz, x=None, y=None, C=1.0, sample=1):
        # loss function
        if x is not None:
            batchsize = len(x.data)
        elif y is not None:
            batchsize = len(y.data)
        else:
            raise ValueError('x or y must be geven')

        rec_loss = 0
        for l in range(sample):
            z = qz.sample()

            nll = 0
            if x is not None:
                nll += self.pxgz(z).nll(x)
            if y is not None:
                nll += self.pygz(z).nll(y)
            
            rec_loss += nll / (sample * batchsize)
            
        kl_loss = rv.gaussian_kl_standard(qz) / batchsize
        loss = rec_loss + C * kl_loss
        return loss

