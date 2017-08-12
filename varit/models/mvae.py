# coding: utf-8

import numpy as np
from scipy.misc import logsumexp
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
import varit.util.random_variables as rv


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

# evaluate

# def conditional_log_likelihood_xgx(vae, x, n_sample=1):
#     batchsize = len(x.data)
    
#     ll_par_x = []
#     for i in range(batchsize):
#         qz = vae.qzgx(x[i:i+1])
#         qzx = vae.qzxgx(x[i:i+1])
        
#         z_sample = qz.sample(n_sample)
#         zx_sample = qzx.sample(n_sample)
#         px = vae.pxgzxz(F.concat([zx_sample, z_sample]))
#         x_ll = px.log_likelihood(F.tile(x[i:i+1], (n_sample, 1)),
#                                  reduce='no')
#         cll = -np.log(n_sample) + logsumexp(F.sum(x_ll, axis=1).data)
#         ll_par_x.append(cll)
#     return np.mean(ll_par_x)

# def conditional_log_likelihood_ygy(vae, y, n_sample=1):
#     batchsize = len(y.data)

#     ll_par_y = []
#     for i in range(batchsize):
#         qz = vae.qzgy(y[i:i+1])
#         qzy = vae.qzygy(y[i:i+1])
        
#         z_sample = qz.sample(n_sample)
#         zy_sample = qzy.sample(n_sample)
#         py = vae.pygzyz(F.concat([zy_sample, z_sample]))
#         y_ll = py.log_likelihood(F.tile(y[i:i+1], (n_sample, 1)),
#                                  reduce='no')
#         cll = -np.log(n_sample) + logsumexp(F.sum(y_ll, axis=1).data)
#         ll_par_y.append(cll)
#     return np.mean(ll_par_y)

# def conditional_log_likelihood_xgy(vae, x, y, pzx, n_sample=1):
#     batchsize = len(x.data)
    
#     ll_par_x = []
#     for i in range(batchsize):
#         qz = vae.qzgy(y[i:i+1])
        
#         z_sample = qz.sample(n_sample)
#         zx_sample = pzx.sample(n_sample)
#         px = vae.pxgzxz(F.concat([zx_sample, z_sample]))
#         x_ll = px.log_likelihood(F.tile(x[i:i+1], (n_sample, 1)),
#                                  reduce='no')
#         cll = -np.log(n_sample) + logsumexp(F.sum(x_ll, axis=1).data)
#         ll_par_x.append(cll)
#     return np.mean(ll_par_x)

# def conditional_log_likelihood_ygx(vae, x, y, pzy, n_sample=1):
#     batchsize = len(y.data)

#     ll_par_y = []
#     for i in range(batchsize):
#         qz = vae.qzgx(x[i:i+1])
        
#         z_sample = qz.sample(n_sample)
#         zy_sample = pzy.sample(n_sample)
#         py = vae.pygzyz(F.concat([zy_sample, z_sample]))
#         y_ll = py.log_likelihood(F.tile(y[i:i+1], (n_sample, 1)),
#                                  reduce='no')
#         cll = -np.log(n_sample) + logsumexp(F.sum(y_ll, axis=1).data)
#         ll_par_y.append(cll)
#     return np.mean(ll_par_y)
