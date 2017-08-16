# coding: utf-8

import chainer
import varit.random_variables as rv


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
        llf = rv.LogLikelihood(self.pxgz, x)
        rec_loss = rv.expectation(qz, llf, sample) / batchsize
        kl_loss = rv.gaussian_kl_standard(qz) / batchsize
        loss = -(rec_loss - C * kl_loss)
        return loss
