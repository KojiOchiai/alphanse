# coding: utf-8

import numpy as np
from scipy.misc import logsumexp
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
import alphanse.util.random_variables as rv


class VAE(chainer.Chain):
    def __init__(self, qzgx, pxgz):
        super().__init__(
            qzgx=qzgx,
            pxgz=pxgz,
        )
    
    def reconstract(self, x, sample=False):
        pz = self.qzgx(x)
        if sample:
            z = pz.sample()
        else:
            z = pz.mu
        px = self.pxgz(z)
        return px

    def lower_bound(self, x, C=1.0, sample=1):
        # loss function
        batchsize = len(x.data)
        qz = self.qzgx(x)
        
        rec_loss = 0
        for l in range(sample):
            z = qz.sample()
            x_nll = self.pxgz(z).nll(x)
            rec_loss += x_nll / (sample * batchsize)
            
        kl_loss = rv.gaussian_kl_standard(qz) / batchsize
        loss = rec_loss + C * kl_loss
        return loss, rec_loss, kl_loss


# evaluate
