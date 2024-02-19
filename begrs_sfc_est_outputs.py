# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:24:25 2021

This script produces the plots for the posterior parameters samples obtained 
with NUTS on the BEGRS surrogate estimation on US macroeconomic data.
(figure 4.a and table 1 in the paper).

@author: Sylvain Barde, University of Kent
"""
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from corner import corner
from begrs import begrs

save = True
color = True
fontSize = 40
save_path_figs = 'figures/sfc_estimates'
save_path_tables = 'tables'
dataPath = 'simData'
modelPath = 'models'
modelTag = 'benchmark_sfc_2000_sobol'
dataTags = ['base', 'new']
empDataLabels = ['SW data','Crisis data']

num_inducing_pts = 250   # Subset of 'non-sparse' inputs
numiter = 100            # Number of epoch iterations
learning_rate = 0.001    # Learning rate
#-----------------------------------------------------------------------------
# Setup latex output formatting
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})

# Setup colors & adapt save folder
if color is True:
    cvecBase = ['r','b']
    save_path_figs += '/color'
else:
    cvecBase = ['gray','silver']
    save_path_figs += '/bw'  

# Create save folder if required
if save is True:
    if not os.path.exists(save_path_figs):
        os.makedirs(save_path_figs,mode=0o777)
        
    if not os.path.exists(save_path_tables):
        os.makedirs(save_path_tables,mode=0o777)

# Labels
variables = [r'\zeta_c', 
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

variableDescr = ['Bank risk aversion (C firms)',
                 'Bank risk aversion (K firms)',
                 'Profit weight in firm inv.',
                 'Capacity weight in firm inv.',
                 'C firm precaution deposits',
                 'Intens. of choice - C/K markets',
                 'Intens. of choice - credit/deposit',
                 'Adaptive expectation param.',
                 'Labour turnover ratio',
                 'Folded normal std. dev.',
                 'Haircut param. - firm default',
                 'Unemp. threshold - wage rev.']

# Original calibration (for table)
params = np.array([
    3.922445,   # P_a: Bank risk aversion (C firms)
    21.513347,  # P_b: Bank risk aversion (K firms)
    0.01,       # P_c: Profit Weight in firm investment
    0.02,       # P_d: Capacity Utilization Weight in firm investment
    1,          # P_e: Cons. Firms Precautionary Deposits
    0.15,       # P_f: Intensity of choice - C/K markets
    0.2,        # P_g: Intensity of choice - credit/deposit markets
    0.25,       # P_h: Adaptive expectation parameter
    0.05,       # P_i: Labour turnover ratio
    0.0094,     # P_j: Folded normal std. dev.
    0.5,        # P_k: Haircut parameter for defaulted firms
    0.08])      # P_l: Unemployment threshold in wage revision

#-----------------------------------------------------------------------------
# Create a begrs estimation object, load existing model
lrnStr = '{:f}'.format(learning_rate).replace('0.','').rstrip('0')
modelDir = modelPath + '/' + modelTag  + \
  '/begrs_ind_{:d}_lr_{:s}_ep_{:d}'.format(num_inducing_pts,lrnStr,numiter)
begrsEst = begrs()
begrsEst.load(modelDir)
parameter_range = begrsEst.parameter_range

# Load empirical samples, plot basic diagnostics
ndim = len(variables)
flatSamples = []
modes = []
means = []
for dataTag in dataTags:
    # Load NUTS samples, get mean and mode
    fil = open(modelDir + '/estimates_{:s}.pkl'.format(dataTag),'rb')
    datas = fil.read()
    fil.close()
    results = pickle.loads(datas,encoding="bytes")
    empSamples = results['samples']
    
    # Append mode/mean/sample to list
    modes.append(results['mode'])
    means.append(np.mean(empSamples, axis = 0))
    flatSamples.append(empSamples)
    #--------------------------------------------------------------------------
    # Trace Plots - Diagnostic (not saved)
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(empSamples)

    # Corner plot - Diagnostic (not saved)
    figure = corner(empSamples, 
                    labels = [r'${:s}$'.format(s) for s in variables])

#-----------------------------------------------------------------------------
# Individual plots
for i in range(ndim):
    x_range = parameter_range[i,1] - parameter_range[i,0]
    xlim_left = parameter_range[i,0] - x_range*0.025
    xlim_right = parameter_range[i,1] + x_range*0.025
    
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(1, 1, 1)
    y_max = 0
    for flatSample, theta_mode, theta_mean, cvec, label in zip(flatSamples,
                                                  modes,
                                                  means,
                                                  cvecBase,
                                                  empDataLabels):
        paramSample = flatSample[:,i]  
        res = ax.hist(x=paramSample, bins='fd', density = True, 
                      edgecolor = 'black', color = cvec, alpha=0.3, 
                      label = label)
        y_max = max(y_max,max(res[0]))
        
    y_max *= 1.25
    ax.plot([params[i],params[i]], [0, y_max], linestyle = 'solid',
            linewidth=1, color = 'k', label = r'Orig')
        
    ax.set_xlabel(r'${:s}$'.format(variables[i]), 
                  fontdict = {'fontsize': fontSize})
    ax.xaxis.set_label_coords(.95, -.075)
    ax.axes.yaxis.set_ticks([])
    ax.legend(loc='best', frameon=False, prop={'size':fontSize})

    ax.set_ylim(top = y_max, bottom = 0)
    ax.set_xlim(left = xlim_left,right = xlim_right)
    ax.plot(xlim_right, 0, ">k", ms=15, clip_on=False)
    ax.plot(xlim_left, y_max, "^k", ms=15, clip_on=False)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params(axis='x', labelsize=fontSize)    
    
    if save is True:
        plt.savefig(save_path_figs + "/dens_{:s}.pdf".format(
            variables[i].replace('\\', '')), format='pdf',bbox_inches='tight')

#-----------------------------------------------------------------------------
# Combined plot
fontSize = 20
subplotSpec = gridspec.GridSpec(ncols=4, nrows=3)
fig = plt.figure(figsize=(16,9))

for i in range(ndim):
    x_range = parameter_range[i,1] - parameter_range[i,0]
    xlim_left = parameter_range[i,0] - x_range*0.025
    xlim_right = parameter_range[i,1] + x_range*0.025
    
    ax = fig.add_subplot(subplotSpec[i])
    y_max = 0
    for flatSample, theta_mode, theta_mean, cvec, label in zip(flatSamples,
                                                  modes,
                                                  means,
                                                  cvecBase,
                                                  empDataLabels):
        paramSample = flatSample[:,i]  
        res = ax.hist(x=paramSample, bins='fd', density = True, 
                      edgecolor = None, color = cvec, alpha=0.4, 
                      label = label)
        y_max = max(y_max,max(res[0]))
        
    y_max *= 1.25
    ax.plot([params[i],params[i]], [0, y_max], linestyle = 'solid',
            linewidth=1, color = 'k', label = r'Orig')
    
    plt.text(0.9,0.9, r'${:s}$'.format(variables[i]), 
             fontsize = fontSize,
             transform=ax.transAxes)
        
    ax.axes.yaxis.set_ticks([])
    ax.set_ylim(top = y_max, bottom = 0)
    ax.set_xlim(left = xlim_left,right = xlim_right)
    ax.plot(xlim_right, 0, ">k", ms=8, clip_on=False)
    ax.plot(xlim_left, y_max, "^k", ms=8, clip_on=False)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params(axis='x', labelsize=fontSize)    

leg = fig.legend(*ax.get_legend_handles_labels(), 
                 loc='lower center', ncol= 3,
                 frameon=False, prop={'size':fontSize})
    
if save is True:
    plt.savefig(save_path_figs + "/dens_all.pdf",
                format='pdf',bbox_inches='tight')

#-----------------------------------------------------------------------------
# Generate Latex table
l1 = max([len(var) for var in variableDescr])
l2 = max([len(var) for var in variables])
l3 = 6

rowStr = ('{:s} {:l1s} & ${:l2s}$ & {:l3.3f} & {:l3g} - {:l3g} & {:l3.3f} & '+
          '{:l3.3f} & {:l3.3f} & {:l3.3f} \\\\')
rowStr = rowStr.replace('l1','{:d}'.format(l1))
rowStr = rowStr.replace('l2','{:d}'.format(l2))
rowStr = rowStr.replace('l3','{:d}'.format(l3))

tableStr = []
tableStr.append('\\begin{tabular}{lcrcrrrr}')
tableStr.append('\\hline')
tableStr.append('\\T  \\B & & &  & \\multicolumn{4}{c}'+
                '{ Posterior estimates } \\\\')
tableStr.append('\\cline{5-8}')
tableStr.append('\\T & & \\multicolumn{1}{c}{Base} & '+
                '\\multicolumn{1}{c}{Prior} & \\multicolumn{2}{c}'+
                '{{ {:s} }} & \\multicolumn{{2}}{{c}}{{ {:s} }} \\\\'.format(
                    empDataLabels[0],empDataLabels[1]))
tableStr.append('\\multicolumn{2}{c}{Parameter} \\B & '+
                '\\multicolumn{1}{c}{value} & \\multicolumn{1}{c}{range} '+
                '& Mode & Mean & Mode & Mean \\\\')
tableStr.append('\\hline')
for i in range(ndim):
    if i == 0:
        pad = '\\T '
    elif i == ndim-1:
        pad = '\\B '
    else:
        pad = '   '
            
    tableStr.append(rowStr.format(pad,
                                  variableDescr[i],
                                  variables[i],
                                  params[i],
                                  parameter_range[i,0],
                                  parameter_range[i,1],
                                  modes[0][i],
                                  means[0][i],
                                  modes[1][i],
                                  means[1][i]))
    
tableStr.append('\\hline')
tableStr.append('\\end{tabular}')

print("\n".join(tableStr))

if save is True:
    with open(save_path_tables + '/table2.txt', 'w') as f_out:
        f_out.write("\n".join(tableStr))
    f_out.close
