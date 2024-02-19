# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 11:11:24 2021

This scripts trains a BEGRS surrogate on the simulated data produced by the 
Caiani et al. (2016) Stock-flow consistent benchmark model.

@author: Sylvain Barde, University of Kent
"""

import numpy as np
import pickle
import zlib
import os

from begrs import begrs

dataPath = 'simData'
modelPath = 'models'
modelTag = 'benchmark_sfc_2000_sobol'
numSamples = 1000
numObs = 200
num_tasks = 7

# Set BEGRS hyper parameters
num_latents = 4          # Identify by pca?
batchSize = 20000        # Size of training minibatches (mainly a speed issue)
num_inducing_pts = 250   # Subset of 'non-sparse' inputs
numiter = 100            # Number of epoch iterations 
learning_rate = 0.001    # Learning rate

# Load samples & parameter ranges
fil = open(dataPath + '/' + modelTag + '/parampool.pkl','rb')
datas = fil.read()
fil.close()
params = pickle.loads(datas,encoding="bytes")
samples = params['samples'][0:numSamples,:]
parameter_range = params['parameter_range']

# Load & repackage simulation data
fil = open(dataPath + '/' + modelTag + '/' + modelTag + '_data.pkl','rb')
datas = zlib.decompress(fil.read(),0)
fil.close()
simData = pickle.loads(datas,encoding="bytes")

modelData = np.zeros([numObs,num_tasks,numSamples])
for i in range(numSamples):
    modelData[:,:,i] = simData[i][0:numObs,:,0]
    
# Create a begrs estimation object, train on simulated data
begrsEst = begrs()
begrsEst.setTrainingData(modelData, samples, parameter_range)
begrsEst.train(num_latents, num_inducing_pts, batchSize, numiter, 
                learning_rate)

# Save trained model
savePath = modelPath + '/' + modelTag
if not os.path.exists(savePath):
    os.makedirs(savePath,mode=0o777)

lrnStr = '{:f}'.format(learning_rate).replace('0.','').rstrip('0')
begrsEst.save(savePath + '/begrs_ind_{:d}_lr_{:s}_ep_{:d}'.format(
                    num_inducing_pts,lrnStr,numiter))

