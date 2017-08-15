# coding: utf-8

import chainer
import chainer.functions as F
import varit.random_variables as rv


class JMVAE(chainer.Chain):
    def __init__(self, qzgxy, qzgx, qzgy, pxgz, pygz):
        super().__init__(
            qzgxy=qzgxy,
            qzgx=qzgx,
            qzgy=qzgy,
            pxgz=pxgz,
            pygz=pygz,
        )
    
    def reconstract(self, x=None, y=None, sample=False):
        if (x is not None)  & (y is not None):
            pz = self.qzgxy(F.concat([x, y]))
        elif x is not None:
            pz = self.qzgx(x)
        elif y is not None:
            pz = self.qzgy(y)
        else:
            raise ValueError('x or y must be geven')
                        
        return self.generate(pz, sample)
                        
    def generate(self, pz, sample=False):
        if sample:
            z = pz.sample()
        else:
            z = pz.mu
        px = self.pxgz(z)
        py = self.pygz(z)
        return px, py

    def lower_bound(self, x, y, C=1.0, alpha=1, sample=1):
        # loss function
        qz = self.qzgxy(F.concat([x, y]))
        vae_loss = self.free_energy(qz, x, y, C=C, sample=sample)
        qzgx = self.qzgx(x)
        qzgy = self.qzgy(y)
        modal_loss = rv.Dkl(qz, qzgx) + rv.Dkl(qz, qzgy)
        return vae_loss + alpha * modal_loss
        
    def free_energy(self, qz, x=None, y=None, C=1.0, sample=1):
        # loss function
        llf = []
        if x is not None:
            batchsize = len(x.data)
            llf.append(rv.LogLikelihood(self.pxgz, x))
        if y is not None:
            batchsize = len(y.data)
            llf.append(rv.LogLikelihood(self.pygz, y))
        if (x is None) and (y is None):
            raise ValueError('x or y must be geven')

        rec_loss = rv.expectation(qz, llf, sample) / batchsize
        kl_loss = rv.gaussian_kl_standard(qz) / batchsize
        loss = -(rec_loss - C * kl_loss)
        return loss
