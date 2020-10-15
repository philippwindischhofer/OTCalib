#!/usr/bin/env python
# coding: utf-8

device='cpu'

import os

# set number of threads to e used by torch
# os.environ["OMP_NUM_THREADS"] = "4"

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

nbatches = 500
nepochs = 1

# some global settings for the logging
from time import gmtime, strftime
time_suffix = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
outprefix = os.path.join('toy_example', time_suffix)

# some global settings for the dataset
length_data = 10000
length_MC = 10000

# some global settings pertaining to the models
#activation = nn.LeakyReLU
activation = nn.Tanh
adversary_lr = 5e-2
transport_lr = 3e-2
lr_decay = 0.94

# some global settings for the training
batchsize = 1024

controlplots(int(1e6))

# -----------------------------------------------
# start preparing the networks
# -----------------------------------------------

width = 5
depth = 5

def init_weights(m):
    if type(m) == nn.Linear:
        #torch.nn.init.xavier_uniform(m.weight)
        abswidth = 4000.0 / np.sqrt((width * depth))
        torch.nn.init.uniform_(m.weight, a = -abswidth, b = abswidth)
        m.bias.data.fill_(0.01)

alldata = genData(length_data, device)
allmc = genMC(length_MC, device)

# transport = fullyConnected(number_layers = 5, number_inputs = 1, number_outputs = 1, hidden_units = 55, activation = activation)
# adversary = fullyConnected(number_layers = 5, number_inputs = 1, number_outputs = 1, hidden_units = 55, activation = activation)

transport = fullyConnected(number_layers = depth, number_inputs = 1, number_outputs = 1, hidden_units = width, activation = activation)
adversary = fullyConnected(number_layers = depth, number_inputs = 1, number_outputs = 1, hidden_units = width, activation = activation)

transport.apply(init_weights)
adversary.apply(init_weights)

transport.to(device)
adversary.to(device)
    
toptim = torch.optim.SGD(transport.parameters(), lr = transport_lr)
aoptim = torch.optim.SGD(adversary.parameters(), lr = adversary_lr)

tsched = torch.optim.lr_scheduler.ExponentialLR(toptim, lr_decay)
asched = torch.optim.lr_scheduler.ExponentialLR(aoptim, lr_decay)

writer = SummaryWriter(outprefix)

# -----------------------------------------------
# start the training
# -----------------------------------------------

for epoch in range(nepochs):
    radvloss = 0
    fadvloss = 0
    tadvloss = 0
    ttransloss = 0
    realavg = 0
    fakeavg = 0

    print("epoch:", epoch)

    for batch in range(nbatches):

        # --------------------
        # adversary update
        # --------------------
        
        for cur in range(1):

            # sample from data and MC
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
            
            # calibrate the MC with the current transport model
            transporting = trans(transport, mc)
            transported = transporting + mc
            
            fake = adversary(transported)
            fakeavg += torch.mean(fake).item()
            
            tmp2 = \
                   binary_cross_entropy_with_logits(
                       fake
                       , torch.zeros_like(real)
                       , reduction='mean'
                   )
            
            fadvloss += tmp2.item()
            
            # total adversary loss
            loss = tmp1 + tmp2
            
            # train the adversary
            loss.backward()
            aoptim.step()

            toptim.zero_grad()
            aoptim.zero_grad()

        # --------------------
        # transport network update
        # --------------------

        # sample from data and MC
        data = alldata[torch.randint(alldata.size()[0], size=(batchsize,), device=device)]
        mc = allmc[torch.randint(allmc.size()[0], size=(batchsize,), device=device)]

        transporting = trans(transport, mc)
        transported = transporting + mc

        fake = adversary(transported)

        tmp2 =\
          binary_cross_entropy_with_logits(
            fake
          , torch.ones_like(real)
          , reduction='mean'
          )

        # train the transport network to force the adversary output on transported MC to be one
        # (i.e. the data label)
        tadvloss += tmp2.item()
        loss = tmp2

        loss.backward()
        toptim.step()

    # update the learning rates
    tsched.step()
    asched.step()

    def get_cur_lr(optim):
        par_groups = optim.param_groups
        for par_group in par_groups:
            if "lr" in par_group:
                return par_group["lr"]
    
    # write tensorboard info once per epoch
    writer.add_scalar('cur_lr', get_cur_lr(toptim), epoch)
    writer.add_scalar('radvloss', radvloss / nbatches, epoch)
    writer.add_scalar('fadvloss', fadvloss / nbatches, epoch)
    writer.add_scalar('tadvloss', tadvloss / nbatches, epoch)
    writer.add_scalar('ttransloss', ttransloss / nbatches, epoch)
    writer.add_scalar('realavg', realavg / nbatches, epoch)
    writer.add_scalar('fakeavg', fakeavg / nbatches, epoch)

    # make debug plots in each epoch to follow along
    detailed_plots(transport, adversary, writer, epoch, device)
    
