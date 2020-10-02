# TODO
# I don't know how many of these imports are strictly necessary.

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from math import e
import otcalibutils


import torch.nn as nn
from collections import OrderedDict

from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.tensorboard import SummaryWriter
from math import log, exp

from itertools import product

def poly(cs, xs):
  ys = torch.zeros_like(xs)
  for (i, c) in enumerate(cs):
    ys += c * xs**i

  return ys


def discMC(xs, logpts):
  return poly([0, 0.6, 1.2, 1.1], xs) * poly([1, 0.1, -0.01], logpts)

def discData(xs, logpts):
  return poly([0, 0.7, 1.1, 1.3], xs) * poly([1, -0.1, 0.02], logpts)

def logptMC(xs):
  return torch.log(poly([25, 200, 7], -torch.log(xs)))

def logptData(xs):
  return torch.log(poly([25, 220, 5], -torch.log(xs)))

def genMC(n, device):
  
  stdev = 1
  mean = -2
  samples = torch.randn(n, device = device) * stdev + mean

  return torch.unsqueeze(samples, 1)

def genData(n, device):

  # from Gaussian
  stdev = 1
  mean = 2
  samples = torch.randn(n, device = device) * stdev + mean

  # # from uniform distribution
  # start = 2
  # end = 4
  # samples = torch.rand(n, device = device) * (end - start) + start

  return torch.unsqueeze(samples, 1)

# give high-pT jets more weight to improve convergence
# similar idea to boosting
def ptWeight(logpts):
    pts = torch.exp(logpts)
    w = torch.exp(pts / e**5.5)
    return w / torch.mean(w)


def histcurve(bins, fills, default):
  xs = [x for x in bins for i in (1, 2)]
  ys = [default] + [fill for fill in fills for i in (1, 2)] + [default]

  return (xs, ys)


def controlplots(n):
  mc = genMC(n, 'cpu').numpy()[:, 0]
  data = genData(n, 'cpu').numpy()[:, 0]

  fig = plt.figure()
  ax = fig.add_subplot(111)
  
  _ = ax.hist([mc, data], bins=25, label=["mc", "data"])
  ax.legend()
  fig.savefig("controlplots.pdf")


def layer(n, m, act):
  return \
    nn.Sequential(
      nn.Linear(n, m)
    , act(inplace=True)
    )


def sequential(xs):
    d = OrderedDict()
    for (i, x) in enumerate(xs):
        d[str(i)] = x

    return nn.Sequential(d)


def fullyConnected(number_layers, number_inputs, number_outputs, hidden_units, activation):
  return \
    nn.Sequential(
      nn.Linear(number_inputs, hidden_units)
      , activation(inplace=True)
      , sequential([layer(hidden_units, hidden_units, activation) for i in range(number_layers)])
      , nn.Linear(hidden_units, number_outputs)
    )


def tloss(xs):
  return torch.mean(xs**2)


def save(path, transport, adversary, toptim, aoptim):
  torch.save(
      { 'transport_state_dict' : transport.state_dict()
      , 'adversary_state_dict' : adversary.state_dict()
      , 'toptim_state_dict' : toptim.state_dict()
      , 'aoptim_state_dict' : aoptim.state_dict()
      }
    , path
  )

def load(path, transport, adversary, toptim, aoptim):
  checkpoint = torch.load(path)
  transport.load_state_dict(checkpoint["transport_state_dict"])
  adversary.load_state_dict(checkpoint["adversary_state_dict"])
  toptim.load_state_dict(checkpoint["toptim_state_dict"])
  aoptim.load_state_dict(checkpoint["aoptim_state_dict"])



def tonp(xs):
  return xs.cpu().detach().numpy()


def detailed_plots(transport, adversary, writer, epoch, device):

  # -----------------------------------------------
  # make closure plot to check that transported MC
  # indeed matches the data
  # -----------------------------------------------

  length_data = 10000
  toy_MC = genMC(length_data, device)
  toy_data = genData(length_data, device)

  transported_MC = tonp(toy_MC + transport(toy_MC))

  fig = plt.figure()
  ax = fig.add_subplot(111)  
  ax.hist([toy_MC[:, 0], toy_data[:, 0], transported_MC[:, 0]], bins=25, label=["mc", "data", "transported mc"])
  ax.legend()

  writer.add_figure("closure", fig, global_step = epoch)
  plt.close()

  # -----------------------------------------------
  # plot the action of the transport function itself
  # -----------------------------------------------

  xvals = torch.unsqueeze(torch.linspace(-4.0, 4.0, 1000), 1)
  yvals = tonp(xvals + transport(xvals))

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(xvals, yvals)

  writer.add_figure("transport function", fig, global_step = epoch)
  plt.close()

  # -----------------------------------------------
  # plot the output delivered by the adversary
  # -----------------------------------------------

  xvals = torch.unsqueeze(torch.linspace(-4.0, 4.0, 1000), 1)
  yvals = tonp(adversary(xvals))

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(xvals, yvals)

  writer.add_figure("adversary output", fig, global_step = epoch)
  plt.close()


def trans(transport, mc):

    return transport(mc)

def ptbin(low, high, samps):
  pts = torch.exp(samps[:,1])
  lowcut = low < pts
  highcut = pts < high

  cut = torch.logical_and(lowcut, highcut)

  return samps[cut]
