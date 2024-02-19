# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 17:16:07 2016

This script produces the MIC comparison for the US estimation. It also 
includes acalculationof the log posterior probability of the surrogate model
(table 2 in the paper).

@author: Sylvain Barde, University of Kent
"""

import numpy as np
import pickle
import os
from begrs import begrs
#-----------------------------------------------------------------------------
# Define posterior based on soft prior and surrogate likelihood
def logP(sample):
        
    prior = begrsEst.softLogPrior(sample)
    logL = begrsEst.logP(sample)

    return (prior[0] + logL[0], prior[1] + logL[1])
#------------------------------------------------------------------------------
# Load/Save directories
load_dir = 'scores'
emp_dir = 'empData'
save_path_tables = 'tables'
save = True

# Identify a begrs estimation object
modelPath = 'models'
modelTag = 'benchmark_sfc_2000_sobol'
num_inducing_pts = 250   # Subset of 'non-sparse' inputs
numiter = 100             # Number of epoch iterations 
learning_rate = 0.001    # Learning rate 

# List datasets
simTags = ['us',                # 0
           'new']               # 1
datasetTags = ['base',              # 0
               'new']               # 1
datasetFiles = ['usdata_base.txt',  # 0
                'usdata_new.txt']   # 1 
datasetLabels = [r'Smets \& Wouters dataset (1965:Q1 - 2004:Q4)',
                 r'Crisis period (1997:Q1 - 2017:Q2)']

# Pick training method from list (2nd is for for robustness checks)
method = 0
lags = [1,3]
methods = ['high_L1',               # 0
           'low_L3']                # 1

# Below, list combinations used for MIC average calculation 
var_vec_base = [[7,6,5,4,3,2,1],    # 0
                [1,7,6,5,4,3,2],    # 1
                [2,1,7,6,5,4,3],    # 2
                [3,2,1,7,6,5,4],    # 3
                [4,3,2,1,7,6,5],    # 4
                [5,4,3,2,1,7,6],    # 5
                [6,5,4,3,2,1,7],    # 6
                
                [1,2,3,4,5,6,7],    # 7
                [7,1,2,3,4,5,6],    # 8
                [6,7,1,2,3,4,5],    # 9
                [5,6,7,1,2,3,4],    # 10
                [4,5,6,7,1,2,3],    # 11
                [3,4,5,6,7,1,2],    # 12
                [2,3,4,5,6,7,1],    # 13
                
                [1,7,2,6,3,5,4],    # 14
                [4,1,7,2,6,3,5],    # 15
                [5,4,1,7,2,6,3],    # 16
                [3,5,4,1,7,2,6],    # 17
                [6,3,5,4,1,7,2],    # 18
                [2,6,3,5,4,1,7],    # 19
                [7,2,6,3,5,4,1],    # 20

                [7,1,6,2,5,3,4],    # 21
                [4,7,1,6,2,5,3],    # 22
                [3,4,7,1,6,2,5],    # 23
                [5,3,4,7,1,6,2],    # 24
                [2,5,3,4,7,1,6],    # 25
                [6,2,5,3,4,7,1],    # 26
                [1,6,2,5,3,4,7],    # 27

                [2,7,6,5,1,4,3],    # 28
                [3,2,7,6,5,1,4],    # 29
                [4,3,2,7,6,5,1],    # 30
                [1,4,3,2,7,6,5],    # 31
                [5,1,4,3,2,7,6],    # 32
                [6,5,1,4,3,2,7],    # 33
                [7,6,5,1,4,3,2]]    # 34

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

# Create a begrs estimation object, load existing model
lrnStr = '{:f}'.format(learning_rate).replace('0.','').rstrip('0')
modelDir = modelPath + '/' + modelTag  + \
  '/begrs_ind_{:d}_lr_{:s}_ep_{:d}'.format(num_inducing_pts,lrnStr,numiter)
begrsEst = begrs()
begrsEst.load(modelDir)

# Iterate over datasets results to get table rows
tableComponents  = []
for dataset in range(2):
    
    # Load empirical samples, get mode and mean parametrisations
    fil = open(modelDir+'/estimates_{:s}.pkl'.format(datasetTags[dataset]),
               'rb')
    datas = fil.read()
    fil.close()
    results = pickle.loads(datas,encoding="bytes")
    theta_mode = results['mode']
    theta_mean = np.mean(results['samples'], axis = 0)
    parametrisations = [params, theta_mode, theta_mean]

    # Generate list of MIC scores to load
    simTag = simTags[dataset]
    models = ['benchmark_sfc_baseline',
              'benchmark_sfc_{:s}_mode'.format(simTag),
              'benchmark_sfc_{:s}_mean'.format(simTag)]
    
    # Load empirical data to size score arrays and use in BEGRS posterior
    emp_path = emp_dir + '//' + datasetFiles[dataset]
    empData = np.loadtxt(emp_path, delimiter="\t") 
    T,num_vars = empData.shape
    T -= lags[method]
    if dataset == 0:
        obs = 160             # Number used in US SW analysis & comparison
        empData = empData[69:-1,:]
    else:
        obs = T               # For crisis period, use all data
        
    begrsEst.setTestingData(empData)
    
    start = T - obs
    table_rows = []
    for model,param in zip(models,parametrisations):
                        
        # - Generate load/save paths
        load_path = load_dir + '//' + simTag + '//' + \
            methods[method] + '//' + model
        
        var_scores = np.zeros([T,num_vars])
        scores_full = np.zeros([T,num_vars])
        table_row = np.zeros(num_vars+2)
        
        for j, var_vec in enumerate(var_vec_base):
        
            num_vars = len(var_vec)
            var_str = ''
            for var_i in var_vec:
                var_str = var_str + str(var_i)   
                
            fil = open(load_path + '//scores_var_' + var_str + '.pkl','rb')
            datas = fil.read()
            fil.close()
            
            results = pickle.loads(datas,encoding="bytes")
            setting_scores = results[1]
            
            # Extract scores from saved raw data
            scores = np.zeros([T, num_vars])
            for k in range(num_vars):
                scores[:,k] = setting_scores[k]   
        
            scores_full += scores/len(var_vec_base)
            
            # Get individual variable scores from last variable in the first 
            # 7 measures
            if j < 7:
                var_scores[:,j] += scores[:,-1]
            
            table_row[0:7] = np.sum(var_scores[start:T,:],0)
        table_row[-2] = sum(np.sum(scores_full[start:T,:],0))
        table_row[-1] = -logP(begrsEst.center(param))[0]
        
            
        table_rows.append(table_row)
    tableComponents.append(np.asarray(table_rows))

#------------------------------------------------------------------------------
# Generate save folder if required
if save is True:        
    if not os.path.exists(save_path_tables):
        os.makedirs(save_path_tables,mode=0o777)

# Iterate over datasets to generate table
rowNames = ['Original', 'Mode', 'Mean' ]
l1 = max([len(var) for var in rowNames])

varName = '{:l1s} '.replace('l1','{:d}'.format(l1))
cellStr = '& {:8.2f} '

tableStr = []
tableStr.append('\\begin{tabular}{lrrrrrrrrr}')
tableStr.append('\\hline')
tableStr.append(r'\B \T & L & r & $\pi$ & $\Delta y$ & $\Delta c$ & ' +
                '$\Delta i$ & $\Delta w$ & Aggr. & -$\ln P$\\')
tableStr.append('\\hline')

for label, table_rows in zip(datasetLabels,tableComponents):
    
    tableStr.append('\\multicolumn{{{:d}}}{{l}}{{\\T\\B \\emph{{{:s}}}}} \\\\'.format(
        num_vars+2,label))
    
    for i, tableRow in enumerate(table_rows):
        if i == 0:
            pad = '\\T '
        elif i == len(table_rows)-1:
            pad = '\\B '
        else:
            pad = '   '
                
        rowStr = pad + varName.format(rowNames[i])
        for cell in tableRow:
            rowStr += cellStr.format(cell)
            
        rowStr += '\\\\'
        tableStr.append(rowStr)
    
tableStr.append('\\hline')
tableStr.append('\\end{tabular}')

if save is True:
    with open(save_path_tables + '/table3.txt', 'w') as f_out:
        f_out.write("\n".join(tableStr))
    f_out.close

print("\n".join(tableStr))
