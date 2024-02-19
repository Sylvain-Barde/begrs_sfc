# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 09:43:00 2023

This script produces the plots for the Simulated Bayesian Computing diagnostics
on the convergence of the posterior obtained via BEGRS when estimating the
Caiani et al. (2016) stock-flow consistent benchmark on US macroeconomic data.
(figure 4.b in the paper).

@author: Sylvain Barde, University of Kent
"""

import os
import pickle
import zlib

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy.stats import binom

# SBC run configurations
save = True
color = True

save_path_figs = 'figures/sbc'
dataPath = 'simData'
sbcPath = 'sbc'
fontSize = 20

modelTag = 'benchmark_sfc_2000_sobol'
num_inducing_pts = 250   
numiter = 100             # Number of epoch iterations (50 init)
learning_rate = 0.001    # Learning rate (0.05 is good here)

# Labels
labels = [r'\zeta_c', 
          r'\zeta_k', 
          r'\gamma_1',             
          r'\gamma_2', 
          r'\sigma',
          r'\varepsilon^{CK}',
          r'\varepsilon^{cd}',
          r'\lambda',
          r'\vartheta',
          r'\sigma^2_{FN}',
          r'\iota',
          r'\nu']

#-----------------------------------------------------------------------------
# Setup latex output and saving folder
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})

# Setup colors & adapt save folder
if color is True:
    cvec = 'b'
    save_path_figs += '/color'
else:
    cvec = 'k'
    save_path_figs += '/bw'

# Create save folder if required
if save is True:
    if not os.path.exists(save_path_figs):
        os.makedirs(save_path_figs,mode=0o777)

#------------------------------------------------------------------------------
# Load SBC results
lrnStr = '{:f}'.format(learning_rate).replace('0.','').rstrip('0')
fil = open(sbcPath + 
           '/{:s}_ind_{:d}_lr_{:s}_ep_{:d}.pkl'.format(modelTag,
                                                       num_inducing_pts,
                                                       lrnStr,
                                                       numiter),
           'rb')
datas = zlib.decompress(fil.read(),0)
fil.close()
sbcData = pickle.loads(datas,encoding="bytes")
hist = sbcData['hist']

#------------------------------------------------------------------------------
# Generate confidence intervaals based on binomial counts.
bins = np.arange(hist.shape[0])
numObs = np.sum(hist,axis = 0)[0]
numBins = hist.shape[0]

pad = 4;
confidenceBoundsX = [-0.5-pad,
                     -0.5-pad/2,
                     -0.5-pad,
                     numBins+0.5+pad,
                     numBins+0.5+pad/2,
                     numBins+0.5+pad]

confidenceBoundsY = [binom.ppf(0.005, numObs, 1/numBins),
                     binom.ppf(0.5, numObs, 1/numBins),
                     binom.ppf(0.995, numObs, 1/numBins),
                     binom.ppf(0.995, numObs, 1/numBins),
                     binom.ppf(0.5, numObs, 1/numBins),
                     binom.ppf(0.005, numObs, 1/numBins)]

x_range = max(confidenceBoundsX) - min(confidenceBoundsX)
xlim_left = min(confidenceBoundsX) - x_range*0.025
xlim_right = max(confidenceBoundsX) + x_range*0.025

# Plot
subplotSpec = gridspec.GridSpec(ncols=4, nrows=3)
fig = plt.figure(figsize=(16,9))

for i in range(hist.shape[1]):
    
    ax = fig.add_subplot(subplotSpec[i])

    confidenceBounds = ax.fill(confidenceBoundsX,
                               confidenceBoundsY,
                               'silver', label = '$95\%$ conf.')
    sbcHist = ax.bar(bins-0.5, 
                 hist[:,i],
                 width=1, 
                 align="edge",
                 edgecolor = None,
                 color = cvec, 
                 alpha = 0.4, 
                 label = '{:d} ind. pts.'.format(num_inducing_pts))
        
    # Set y axis limits
    yMin, yMax = ax.axes.get_ylim()
    yMax *= 1.25

    # Annotate
    plt.text(0.9,0.9, r'${:s}$'.format(labels[i]), 
             fontsize = fontSize,
             transform=ax.transAxes)

    ax.set_ylabel(''),
    ax.set_xlabel('')
    ax.axes.yaxis.set_ticks([])

    ax.set_ylim(top = yMax, bottom = 0)
    ax.set_xlim(left = xlim_left,right = xlim_right)
    ax.plot(xlim_right, 0, ">k", ms=8, clip_on=False)
    ax.plot(xlim_left, yMax, "^k", ms=8, clip_on=False)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params(axis='x', labelsize=fontSize)    
    ax.tick_params(axis='y', labelsize=fontSize) 
        
leg = fig.legend(*ax.get_legend_handles_labels(), 
                 loc='lower center', ncol= 2,
                 frameon=False, prop={'size':fontSize})
        
if save is True:
    plt.savefig(save_path_figs +  
                '/{:s}_ind_{:d}_lr_{:s}_ep_{:d}.pdf'.format(modelTag,
                                                 num_inducing_pts,
                                                 lrnStr,
                                                 numiter), 
                format = 'pdf',bbox_inches='tight')