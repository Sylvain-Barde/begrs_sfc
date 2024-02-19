# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 13:25:18 2021

This script estimates parameter values for the SFC model on US macroeconomic 
data, using a trained BEGRS surrograte model.

@author: Sylvain Barde, University of Kent
"""

import numpy as np
import pickle
from begrs import begrs, begrsNutsSampler

#-----------------------------------------------------------------------------
# Define posterior based on prior and surrogate likelihood
def logP(sample):
        
    prior = begrsEst.softLogPrior(sample)
    logL = begrsEst.logP(sample)

    return (prior[0] + logL[0], prior[1] + logL[1])

#-----------------------------------------------------------------------------
# Create a begrs estimation object, load existing model
modelPath = 'models'
modelTag = 'benchmark_sfc_2000_sobol'
num_inducing_pts = 250   # Subset of 'non-sparse' inputs - 500 or 1000
# numiter = 50             # Number of epoch iterations (50 init)
# learning_rate = 0.025    # Learning rate (0.05 is good here)
numiter = 100             # Number of epoch iterations (50 init)
learning_rate = 0.001    # Learning rate (0.05 is good here)
    
lrnStr = '{:f}'.format(learning_rate).replace('0.','').rstrip('0')
modelDir = modelPath + '/' + modelTag  + \
  '/begrs_ind_{:d}_lr_{:s}_ep_{:d}'.format(num_inducing_pts,lrnStr,numiter)
begrsEst = begrs()
begrsEst.load(modelDir)

#-----------------------------------------------------------------------------
# Estimate on empirical data (compare to original vector)
dataset = 1
datasetFiles = ['usdata_base.txt',  # 0
                'usdata_new.txt']   # 1

datasetTags = ['base',              # 0
               'new']               # 1

emp_path = 'empData//{:s}'.format(datasetFiles[dataset])
xEmp = np.loadtxt(emp_path, delimiter="\t") 
if dataset == 0:
    xEmp = xEmp[69:-1,:]
    
print(' Empirical setting: {:s}'.format(emp_path))

# Create & configure NUTS sampler for BEGRS
posteriorSampler = begrsNutsSampler(begrsEst, logP)
init = np.zeros(begrsEst.num_param)
posteriorSampler.setup(xEmp, init)

# Run and save results
N = 10000
burn = 100
posteriorSamples = posteriorSampler.run(N, burn)
sampleESS = posteriorSampler.minESS(posteriorSamples)
print('Minimal sample ESS: {:.2f}'.format(sampleESS))

results = {'mode' : begrsEst.uncenter(posteriorSampler.mode),
           'samples': posteriorSamples,
           'ess': sampleESS}
fil = open(modelDir + '/estimates_{:s}.pkl'.format(datasetTags[dataset]),'wb')
fil.write(pickle.dumps(results, protocol=2))
fil.close()
#-----------------------------------------------------------------------------