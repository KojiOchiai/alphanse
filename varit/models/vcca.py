# coding: utf-8

import numpy as np
from scipy.misc import logsumexp
import chainer
import chainer.functions as F
import varit.random_variables as rv


class VCCA(chainer.Chain):
    def __init__(self, qzxgx, qzgx, qzgy, qzygy, pxgzxz, pygzyz):
        super().__init__(
            qzxgx=qzxgx,
            qzgx=qzgx,
            qzgy=qzgy,
            qzygy=qzygy,
            pxgzxz=pxgzxz,
            pygzyz=pygzyz,
        )
    
    def reconstract_multiple(self, x, y, share_source='x', sample=False):
        pzx = self.qzxgx(x)
        if share_source is 'x':
            pzshare = self.qzgx(x)
        elif share_source is 'y':
            pzshare = self.qzgy(y)
        pzy = self.qzygy(y)
        return self.generate(pzx, pzshare, pzy, sample)
                        
    def generate(self, pzx, pzshare, pzy, sample=False):
        if sample:
            zx = pzx.sample()
            z_share = self.pzshare.sample()
            zy = pzy.sample()
        else:
            zx = pzx.mu
            z_share = pzshare.mu
            zy = pzy.mu

        px = self.pxgzxz(F.concat([zx, z_share]))
        py = self.pygzyz(F.concat([zy, z_share]))

        return px, py

    def lower_bound(self, x, y, share_source='x', C=1.0, sample=1):
        # loss function
        batchsize = len(x.data)
        
        qzx = self.qzxgx(x)
        if share_source == 'x':
            qzshare = self.qzgx(x)
        if share_source == 'y':
            qzshare = self.qzgy(y)
        qzy = self.qzygy(y)
        qzxz = rv.Concat([qzx, qzshare])
        qzyz = rv.Concat([qzy, qzshare])

        llf_x = rv.LogLikelihood(self.pxgzxz, x)
        llf_y = rv.LogLikelihood(self.pygzyz, y)

        rec_loss = (rv.expectation(qzxz, llf_x, sample)
                    + rv.expectation(qzyz, llf_y, sample)) / batchsize
        kl_loss = (rv.gaussian_kl_standard(qzx)
                  + rv.gaussian_kl_standard(qzshare)
                  + rv.gaussian_kl_standard(qzy)) / batchsize
        loss = -(rec_loss - C * kl_loss)
        return loss, qzshare

    def bi_lower_bound(self, x, y, weight=0.5, C=1.0, sample=1):
        # loss function
        lbx, _ = self.lower_bound(x, y, share_source='x',
                               C=C, sample=sample)
        lby, _ = self.lower_bound(x, y, share_source='y',
                               C=C, sample=sample)
        loss = weight * lbx + (1 - weight) * lby
        return loss

    def bi_lower_bound_kl(self, x, y, weight=0.5, C=1.0,
                          kl_weight=0.001, sample=1):
        # loss function
        lbx, qzgx = self.lower_bound(x, y, share_source='x',
                                     C=C, sample=sample)
        lby, qzgy = self.lower_bound(x, y, share_source='y',
                                     C=C, sample=sample)
        klzz = rv.gaussian_kl(qzgx, qzgy)
        loss = weight * lbx + (1 - weight) * lby + kl_weight * klzz
        return loss

# evaluate

def conditional_log_likelihood_xgx(vae, x, n_sample=1):
    batchsize = len(x.data)
    
    ll_par_x = []
    for i in range(batchsize):
        qz = vae.qzgx(x[i:i+1])
        qzx = vae.qzxgx(x[i:i+1])
        
        z_sample = qz.sample(n_sample)
        zx_sample = qzx.sample(n_sample)
        px = vae.pxgzxz(F.concat([zx_sample, z_sample]))
        x_ll = px.log_likelihood(F.tile(x[i:i+1], (n_sample, 1)),
                                 reduce='no')
        cll = -np.log(n_sample) + logsumexp(F.sum(x_ll, axis=1).data)
        ll_par_x.append(cll)
    return np.mean(ll_par_x)

def conditional_log_likelihood_ygy(vae, y, n_sample=1):
    batchsize = len(y.data)

    ll_par_y = []
    for i in range(batchsize):
        qz = vae.qzgy(y[i:i+1])
        qzy = vae.qzygy(y[i:i+1])
        
        z_sample = qz.sample(n_sample)
        zy_sample = qzy.sample(n_sample)
        py = vae.pygzyz(F.concat([zy_sample, z_sample]))
        y_ll = py.log_likelihood(F.tile(y[i:i+1], (n_sample, 1)),
                                 reduce='no')
        cll = -np.log(n_sample) + logsumexp(F.sum(y_ll, axis=1).data)
        ll_par_y.append(cll)
    return np.mean(ll_par_y)

def conditional_log_likelihood_xgy(vae, x, y, pzx, n_sample=1):
    batchsize = len(x.data)
    
    ll_par_x = []
    for i in range(batchsize):
        qz = vae.qzgy(y[i:i+1])
        
        z_sample = qz.sample(n_sample)
        zx_sample = pzx.sample(n_sample)
        px = vae.pxgzxz(F.concat([zx_sample, z_sample]))
        x_ll = px.log_likelihood(F.tile(x[i:i+1], (n_sample, 1)),
                                 reduce='no')
        cll = -np.log(n_sample) + logsumexp(F.sum(x_ll, axis=1).data)
        ll_par_x.append(cll)
    return np.mean(ll_par_x)

def conditional_log_likelihood_ygx(vae, x, y, pzy, n_sample=1):
    batchsize = len(y.data)

    ll_par_y = []
    for i in range(batchsize):
        qz = vae.qzgx(x[i:i+1])
        
        z_sample = qz.sample(n_sample)
        zy_sample = pzy.sample(n_sample)
        py = vae.pygzyz(F.concat([zy_sample, z_sample]))
        y_ll = py.log_likelihood(F.tile(y[i:i+1], (n_sample, 1)),
                                 reduce='no')
        cll = -np.log(n_sample) + logsumexp(F.sum(y_ll, axis=1).data)
        ll_par_y.append(cll)
    return np.mean(ll_par_y)
