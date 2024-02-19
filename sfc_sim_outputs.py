# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 09:22:07 2024

This script produces the density plots for the empirical obesrvables produced 
by the SFC simulations (figure 5 in the paper).

@author: Sylvain Barde, University of Kent
"""

import os
import pickle
import zlib

import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import iqr
from matplotlib import pyplot as plt
from matplotlib import gridspec
#------------------------------------------------------------------------------
def bandwidth(vec):
    
    # Return KDE bandwidth (Silverman's rule of thumb)
    N = len(vec)
    m = min(np.std(vec), iqr(vec)/1.34)
    
    return 0.9*m*N**(-1/5)
#------------------------------------------------------------------------------
# Setup latex output and saving folder
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})

# Load directories
sim_dir = 'simData'
emp_dir = 'empData'
save_path_figs = 'figures/simulation_visualisation'

# Options
save = True
color = True
tickFontSize = 25
labelFontSize = 35

# List basic labelsand files needed
datasetTags = ['us',                # 0
               'new']               # 1
datasetFiles = ['usdata_base.txt',  # 0
                'usdata_new.txt']   # 1 

modelBaseline = 'benchmark_sfc_baseline'
baselineLabel = 'Baseline params.'
empDataLabels = ['SW data','Crisis data']
modelEstLabels = ['SW est.','Crisis est.']

# Setup colors & line styles
lineStyleBase = ['solid', (0, (5, 10))]
if color is True:
    cvecBase = ['r','b']
    save_path_figs += '/color'
else:
    cvecBase = ['gray','silver']
    save_path_figs += '/bw'  

# Build list of simulated models to examine & formats for outputs
models = [modelBaseline]
modelLabels = [baselineLabel]
colorVec = ['k']
styleVec = ['solid']
for datasetTag, modelEstLabel, color in zip(datasetTags, 
                                            modelEstLabels,
                                            cvecBase):
    models += ['benchmark_sfc_{:s}_mode'.format(datasetTag),
              'benchmark_sfc_{:s}_mean'.format(datasetTag)]
    modelLabels += ['{:s}, mode'.format(modelEstLabel),
                    '{:s}, mean'.format(modelEstLabel)]
    colorVec += [color,color]
    styleVec += lineStyleBase

# Load simulated data forlist of models
simData = []
for model in models:
    sim_path = sim_dir + '//' + model + '//' + model + '_data.pkl'
    fil = open(sim_path,'rb')
    datas = zlib.decompress(fil.read(),0)
    fil.close()
    simData.append(pickle.loads(datas,encoding="bytes"))

# Load empirical data
empDataVec = []
for dataset in range(len(datasetFiles)):

    empPath = emp_dir + '//' + datasetFiles[dataset]
    empDataRaw = np.loadtxt(empPath, delimiter="\t") 
    T,num_vars = empDataRaw.shape
    
    if dataset == 0:
        obs = 160             # Number used in US SW analysis & comparison
    else:
        obs = T               # For EU, usa all
    
    start = T - obs
    empDataVec.append(empDataRaw[start:T,:])

varLabels = ['L',
             'r',
             r'$\pi$',
             r'$\Delta y$',
             r'$\Delta c$',
             r'$\Delta i$',
             r'$\Delta w$']

# Create save folder if required
if save is True:
    if not os.path.exists(save_path_figs):
        os.makedirs(save_path_figs,mode=0o777)

# Iterate over variables to create plots (+ save if required)
for varIndex, label in enumerate(varLabels):
    print(' Running variable {:s}'.format(label))

    kdeVec = []
    xMin = 0
    xMax = 0
    for modelInd in range(len(models)):
        print(' Getting KDE for model {:s}'.format(models[modelInd]))

        for iterInd in range(1000):
            if iterInd == 0:
                simVec = simData[modelInd][iterInd][:,varIndex,:].flatten()
            else:
                simVecIter = simData[modelInd][iterInd][:,varIndex,:].flatten()
                simVec = np.concatenate((simVec,simVecIter))

        xMin = min(xMin, min(simVec))
        xMax = max(xMax, max(simVec))
        Bsim = bandwidth(simVec)
        kdeSim = KernelDensity(bandwidth=Bsim, kernel='gaussian')
        kdeSim.fit(simVec.reshape(-1, 1))
        kdeVec.append(kdeSim)
    
    # Get empirical data and work out plot bounds from there
    empVecs = []
    empMin = 10e6
    empMax = 0
    for empData in empDataVec:
        empVec = empData[:,varIndex]
        empMin = min(empMin, min(empVec))
        empMax = max(empMax, max(empVec))
        empVecs.append(empVec)

    empRange = empMax - empMin
    empMin -= 0.125*empRange
    empMax += 0.125*empRange
    
    # Generate figure
    xPlot = np.linspace(empMin, empMax, 500)
    
    fig = plt.figure(figsize=(16,12))
    subplotSpec = gridspec.GridSpec(ncols=1, nrows=2,
                             hspace=0, 
                             height_ratios=[1, 6])
    
    # Top plot - plot empirical data, store plot handles for legend
    ax0 = fig.add_subplot(subplotSpec[0])
    boxHandles = []
    ind = 1
    for empVarData, color in zip(empVecs, cvecBase):
        bp = ax0.boxplot(empVarData, 
                         positions = [ind],
                         notch ='True', 
                         vert = 0,
                         widths = 0.5,
                         patch_artist=True,
                         boxprops=dict(facecolor=color,
                                       alpha = 0.5))
        boxHandles.append(bp)
        ind -= 1
    
    ax0.axes.yaxis.set_ticks([])
    ax0.set_xlim(left = empMin,right = empMax)
    yMin, yMax = ax0.axes.get_ylim()
    ax0.spines['top'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.plot(empMin, yMax, "^k", ms=15, clip_on=False)
    
    # Bottom plot - Simulated data KDEs
    ax1 = fig.add_subplot(subplotSpec[1])
    plotHandles = []
    yMax = 0
    for kdeSim, modLabel, style, color in zip(kdeVec,
                                       modelLabels,
                                       styleVec,
                                       colorVec):
        logDensSim = kdeSim.score_samples(xPlot.reshape(-1, 1))
        yMax = max(yMax, max(np.exp(logDensSim)))
        ph = ax1.plot(xPlot,np.exp(logDensSim),
                      linestyle = style,
                      color = color,
                      label = modLabel)
        plotHandles.append(ph)

    ax1.legend([handle["boxes"][0] for handle in boxHandles] + 
               [handle[0] for handle in plotHandles],
               empDataLabels + modelLabels,
               frameon=False, prop={'size':labelFontSize})
        
    ax1.set_xlim(left = empMin,right = empMax)
    ax1.set_xlabel(r'${:s}$'.format(label), 
                   fontdict = {'fontsize': labelFontSize})
    ax1.xaxis.set_label_coords(.95, -.075)
    ax1.plot(empMax, 0, ">k", ms=15, clip_on=False)
    ax1.tick_params(axis='x', labelsize = tickFontSize)    
    ax1.set_ylim(bottom = 0,top = yMax*1.25)
    ax1.axes.yaxis.set_ticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    if save is True:

        savelabel = label.replace(' ', '_')
        savelabel = savelabel.replace('\\','')
        savelabel = savelabel.replace('$','')
        
        plt.savefig(save_path_figs + "/dens_{:s}.pdf".format(
            savelabel), format = 'pdf',bbox_inches='tight')
