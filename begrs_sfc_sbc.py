# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 09:43:00 2023

This script runs a simulated Bayesian computing diagnostic on a trained BEGRS 
model on the full simulated SFC testing data.

@author: Sylvain Barde, University of Kent
"""

import numpy as np
import pickle
import zlib
from begrs import begrs, begrsNutsSampler, begrsSbc

#-----------------------------------------------------------------------------
# Define posterior based on prior and surrogate likelihood
def logP(sample):
    
    prior = begrsEst.softLogPrior(sample)
    logL = begrsEst.logP(sample)

    return (prior[0] + logL[0], prior[1] + logL[1])

#-----------------------------------------------------------------------------
# Run configurations
dataPath = 'simData'
modelPath = 'models'
savePath = 'sbc'
numTrainingSamples = 1000
numSbcSamples = 1000
numObs = 200
num_tasks = 7

# Load BEGRS model
modelTag = 'benchmark_sfc_2000_sobol'
num_inducing_pts = 250   
numiter = 100             # Number of epoch iterations 
learning_rate = 0.001     # Learning rate 

lrnStr = '{:f}'.format(learning_rate).replace('0.','').rstrip('0')
modelDir = modelPath + '/' + modelTag  + \
  '/begrs_ind_{:d}_lr_{:s}_ep_{:d}'.format(num_inducing_pts,lrnStr,numiter)
begrsEst = begrs()
begrsEst.load(modelDir)

# Load SBC testing data
# Load samples & parameter ranges
fil = open(dataPath + '/' + modelTag + '/parampool.pkl','rb')
datas = fil.read()
fil.close()
params = pickle.loads(datas,encoding="bytes")
testSamples = params['samples'][numTrainingSamples:numTrainingSamples+numSbcSamples,:]

# Load & repackage simulation data
fil = open(dataPath + '/' + modelTag + '/' + modelTag + '_data.pkl','rb')
datas = zlib.decompress(fil.read(),0)
fil.close()
simData = pickle.loads(datas,encoding="bytes")

testData = np.zeros([numObs,num_tasks,numSbcSamples])
for i in range(numSbcSamples):
    testData[:,:,i] = simData[numTrainingSamples+i][0:numObs,:,0]

# Create SBC object, load data & sampler
SBC = begrsSbc()
SBC.setTestData(testSamples,testData)
SBC.setPosteriorSampler(begrsNutsSampler(begrsEst, logP))

# Run and save results
N = 199             # Number of draws - aim is for histogram to have 50 bins
burn = 150          # Burn-in period to discard
init = np.zeros(begrsEst.num_param)
SBC.run(N, burn, init, essCutoff = 10)  # SBC will auto-thin to produce N-burn ranks
SBC.saveData(savePath + 
              '/{:s}_ind_{:d}_lr_{:s}_ep_{:d}.pkl'.format(modelTag,
                                                          num_inducing_pts,
                                                          lrnStr,
                                                          numiter)) 
#-----------------------------------------------------------------------------