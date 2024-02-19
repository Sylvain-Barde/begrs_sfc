# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 13:26:41 2017

This script produces the simulated SFC training data required to run the 
BEGRS analysis in section 4.2 of the paper.

NOTE: Due to the time-consuming nature of the SFC model, the script uses the 
multiprocessing package to parallelize the simulations over a large number of 
cores. Users should ensure they have access to the required compoute resources 
before running the script.

@author: Sylvain Barde, University of Kent
"""

import multiprocessing as mp
import numpy as np
import time
import os
import pickle
import zlib

from sfc_functions import get_sobol_samples
from sfc_functions import convert_sim_data, process_sim_data, runSFC
#------------------------------------------------------------------------------
if __name__ == '__main__':
    
    # Full run
    numCores = 36
    modelTag = 'benchmark_sfc_2000_sobol'
    setupPath = 'setup/benchmark_sfc'
    javaFiles = ['benchmark_sfc.jar',
                 'benchmark.Main']
    
    # Set parameters for sampling design.
    numSamples = 2000
    skip = 500
    numEval = 0
    numReps = 1
    numSims = 1
    numBurn = 300
    numObs = 501
    monthToQuarter = False
       
    # Set parameter ranges
    tokens = ['P_a','P_b','P_c','P_d','P_e','P_f',
              'P_g','P_h','P_i','P_j','P_k','P_l']
    parameter_range = np.array([
        (1, 10),        # P_a: Bank risk aversion (C firms)
        (5, 40),        # P_b: Bank risk aversion (K firms)
        (0.01, 0.04),   # P_c: Profit Weight in firm investment
        (0.01, 0.04),   # P_d: Capacity Utilization Weight in firm investment
        (0.5, 1.5),     # P_e: Cons. Firms Precautionary Deposits
        (0.05, 0.3),    # P_f: Intensity of choice - C/K markets
        (0.05, 0.3),    # P_g: Intensity of choice - credit/deposit markets
        (0.1, 0.8),     # P_h: Adaptive expectation parameter
        (0.025, 0.15),  # P_i: Labour turnover ratio
        (0.005, 0.015), # P_j: Folded normal std. dev.
        (0.3, 0.7),     # P_k: Haircut parameter for defaulted firms
        (0.05, 0.11)])  # P_l: Unemployment threshold in wage revision
        
    # Create logging directory
    log_path = "logs//" + modelTag
    print('Saving logs to: ' + log_path)
    os.makedirs(log_path,mode=0o777)
    
    # Create working directory
    sim_path = "simData//" + modelTag
    if not os.path.exists(sim_path):
            os.makedirs(sim_path,mode=0o777)
    
    # Create parametrisation
    print("Generating parameter samples for " + modelTag)
    
    param_dims = parameter_range.shape[0]
    samples = get_sobol_samples(numSamples, parameter_range, skip)
    params = {'numObs' : numObs,
              'numBurn': numBurn,
              'numReps' : numReps,
              'numSims' : numSims,
              'samples' : samples,
              'parameter_range' : parameter_range}
        
    fil = open(sim_path + '/parampool.pkl','wb')
    fil.write(pickle.dumps(params, protocol=2))
    fil.close()
    
    # Populate settings                
    settings = {'modelTag':sim_path,
                'tokens':tokens,
                'log_path':log_path,
                'setup_path':setupPath,
                'javaFiles':javaFiles,
                'numReps':numReps,
                'numObs':numObs,
                'numSims':numSims,
                'numCores':numCores,
                'numEval':0,
                'iter_count':0}
    
    # ------------------------------------------------------------------------
    t_start = time.time()
    
    # Populate job
    job_inputs = []
    numTasks = numSamples*numReps
    for i in range(numSamples):
        for j in range(numReps):
            job_inputs.append((settings, samples, numEval + i, j))

    # -- Initialise Display
    print(' ')
    print('+------------------------------------------------------+')
    print('|    Parallel Java run of SFC model - SOBOL sampling   |')
    print('+------------------------------------------------------+')
    print(' Model: ' + modelTag )
    print(' Number of cores: ' + str(numCores) + \
        ' - Number of tasks: ' + str(numTasks))
    print('+------------------------------------------------------+')
    
    # Create pool and run parallel job
    pool = mp.Pool(processes=numCores)
    res = pool.map(runSFC, job_inputs)

    # Close pool when job is done
    pool.close()
        
    # Extract results for timer and process files
    sum_time = 0
    simDataFull = {}
    for i in range(numSamples):
        
        simDataRaw = convert_sim_data(sim_path, numEval + i, numReps)
        
        # Process simulated data to observables, save
        simData = process_sim_data(simDataRaw,
                                   numBurn,
                                   numObs,
                                   monthToQuarter)
        simDataFull[i] = simData

    fil = open(sim_path + '/' + modelTag + '_data.pkl','wb') 
    fil.write(zlib.compress(pickle.dumps(simDataFull, protocol=2)))
    fil.close()

    print('+------------------------------------------------------+')
    timer_1 = time.time() - t_start
    print(' Total running time:     {:10.4f} secs.'.format(timer_1))
    print(' Sum of iteration times: {:10.4f} secs.'.format(sum_time))
    print(' Mean iteration time:    {:10.4f} secs.'.format(
            sum_time/numTasks))
    print(' Speed up:               {:10.4f}'.format(sum_time/timer_1))
    print('+------------------------------------------------------+')
#------------------------------------------------------------------------------