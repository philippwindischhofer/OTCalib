#!/usr/bin/env python
# coding: utf-8


outprefix = 'test/'
device='cuda'

# TODO
# I don't know how many of these imports are strictly necessary.

import numpy as np
import matplotlib    
matplotlib.use('Agg')    
import matplotlib.pyplot as plt    
import torch
from math import e


import torch.nn as nn
from collections import OrderedDict

from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.tensorboard import SummaryWriter
from math import log, exp

from itertools import product

from otcalibutils import *

# toys for validation samples
nval = 2**12
valtoys = torch.rand(nval, device=device)


nbatches = 2**10
nepochs = 200


decays = [0.95]
acts = [("lrelu", nn.LeakyReLU)] #, ("sig", nn.Sigmoid), ("tanh", nn.Tanh)]
bss = [64]
npss = [1]
nlayer = [4]
latent = [256]
lrs = [(1, 1e-1)]
dss = [int(1e5), int(1e7)]

controlplots(int(1e6))
plt.savefig("controlplots.pdf")

for (decay, (actname, activation), batchsize, nps, nlay, nlat, (alr, tlr), datasize) \
  in product(decays, acts, bss, npss, nlayer, latent, lrs, dss):


    alldata = genData(datasize, device)
    allmc = genMC(4*datasize, device)

    transport = fullyConnected(nlay, 2, nlat, 1+nps, activation)

    adversary = fullyConnected(nlay, 2, nlat*2, 1, nn.LeakyReLU)


    transport.to(device)
    adversary.to(device)
    
    toptim = torch.optim.SGD(transport.parameters(), lr=tlr)
    aoptim = torch.optim.SGD(adversary.parameters(), lr=alr)

    tsched = torch.optim.lr_scheduler.ExponentialLR(toptim, decay)
    asched = torch.optim.lr_scheduler.ExponentialLR(aoptim, decay)


    name = \
      "sgdexp_%.2f_act_%s_batch_%d_nps_%d_layers_%d_latent_%d_tlr_%0.2e_alr_%0.2e_datasize_%d" \
        % (decay, actname, batchsize, nps, nlay, nlat, tlr, alr, datasize)

    writer = SummaryWriter(outprefix + name)


    for epoch in range(nepochs):
      radvloss = 0
      fadvloss = 0
      tadvloss = 0
      ttransloss = 0
      realavg = 0
      fakeavg = 0

      print("epoch:", epoch)

      for batch in range(nbatches):

        data = alldata[torch.randint(alldata.size()[0], size=(batchsize,), device=device)]

        mc = allmc[torch.randint(allmc.size()[0], size=(batchsize,), device=device)]

        toptim.zero_grad()
        aoptim.zero_grad()

        real = adversary(data)
        realavg += torch.mean(real).item()

        tmp1 = \
          binary_cross_entropy_with_logits(
              real
            , torch.ones_like(real)
            , reduction='mean'
            )

        radvloss += tmp1.item()


        # add gradient regularization
        # grad_params = torch.autograd.grad(tmp1, adversary.parameters(), create_graph=True, retain_graph=True)
        # grad_norm = 0
        # for grad in grad_params:
        #     grad_norm += grad.pow(2).sum()
        # grad_norm = grad_norm.sqrt()


        thetas = torch.randn((batchsize, nps), device=device)
        transporting = trans(transport, mc, thetas)

        transported = transporting + mc[:,0:1]

        fake = adversary(torch.cat([transported, mc[:,1:]], axis=1))

        fakeavg += torch.mean(fake).item()

        tmp2 = \
          binary_cross_entropy_with_logits(
              fake
            , torch.zeros_like(real)
            , reduction='mean'
            )

        fadvloss += tmp2.item()

        loss = tmp1 + tmp2 # + 0.1*grad_norm

        loss.backward()
        aoptim.step()


        toptim.zero_grad()
        aoptim.zero_grad()

        thetas = torch.randn((batchsize, nps), device=device)
        transporting = trans(transport, mc, thetas)

        transported = transporting + mc[:,0:1]
        fake = adversary(torch.cat([transported, mc[:,1:]], axis=1))

        tmp1 = tloss(transporting)
        ttransloss += tmp1.item()

        tmp2 =\
          binary_cross_entropy_with_logits(
            fake
          , torch.ones_like(real)
          , reduction='mean'
          )

        tadvloss += tmp2.item()

        loss = tmp2 # tmp1 + tmp2

        loss.backward()
        toptim.step()


      tsched.step()
      asched.step()

      # write tensorboard info once per epoch
      writer.add_scalar('radvloss', radvloss / nbatches, epoch)
      writer.add_scalar('fadvloss', fadvloss / nbatches, epoch)
      writer.add_scalar('tadvloss', tadvloss / nbatches, epoch)
      writer.add_scalar('ttransloss', ttransloss / nbatches, epoch)
      writer.add_scalar('realavg', realavg / nbatches, epoch)
      writer.add_scalar('fakeavg', fakeavg / nbatches, epoch)


      # make validation plots once per epoch
      plotPtTheta(transport, 25, valtoys, nps, writer, "pt25", epoch, device)

      plotPtTheta(transport, 50, valtoys, nps, writer, "pt50", epoch, device)

      plotPtTheta(transport, 100, valtoys, nps, writer, "pt100", epoch, device)

      plotPtTheta(transport, 250, valtoys, nps, writer, "pt250", epoch, device)

      plotPtTheta(transport, 500, valtoys, nps, writer, "pt500", epoch, device)

      save(outprefix + name + ".pth", transport, adversary, toptim, aoptim)