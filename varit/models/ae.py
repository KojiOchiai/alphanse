# coding: utf-8

import chainer
import chainer.functions as F


class AE(chainer.Chain):
    def __init__(self, encoder, decoder, loss_func=F.mean_squared_error):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
        )
        self.loss_func = loss_func

    def reconstract(self, x):
        h = self.encoder(x)
        x_hat = self.decoder(h)
        return x_hat

    def loss(self, x, y=None):
        y = y or x
        loss = self.loss_func(self.reconstract(x), y)
        return loss
