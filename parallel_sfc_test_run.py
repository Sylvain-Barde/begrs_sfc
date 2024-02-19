# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 13:26:41 2017

This script produces the simulated SFC data required to run the 
MIC comparison of the parameters estimates obtained by BEGRS against the 
original calibration, in section 4.2 of the paper.

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

from sfc_functions import convert_sim_data, process_sim_data, runSFC
#------------------------------------------------------------------------------
if __name__ == '__main__':
    
    # Full simulation run on single parametrisation
    numCores = 50
    modelTag = 'benchmark_sfc_2000_sobol/begrs_ind_250_lr_001_ep_100'
    setupPath = 'setup/benchmark_sfc'
    javaFiles = ['benchmark_sfc.jar',
                 'benchmark.Main']
    runModel = 'mod5'   
    

    parametrisationDict = {
        'mod1':{'simTag':'benchmark_sfc_baseline',
                'estimatesTag':None,
                'baseline':True,
                'mean':False},
        'mod2':{'simTag':'benchmark_sfc_us_mode',
                'estimatesTag':'estimates_base',
                'baseline':False,
                'mean':False},
        'mod3':{'simTag':'benchmark_sfc_us_mean',
                'estimatesTag':'estimates_base',
                'baseline':False,
                'mean':True},
        'mod4':{'simTag':'benchmark_sfc_new_mode',
                'estimatesTag':'estimates_new',
                'baseline':False,
                'mean':False},
        'mod5':{'simTag':'benchmark_sfc_new_mean',
                'estimatesTag':'estimates_new',
                'baseline':False,
                'mean':True}
        }


    # Set overall parameters for the simulation (same for all)
    numSamples = 1000
    numReps = 1
    numSims = 2
    numEval = 0
    numBurn = 300
    numObs = 551
    monthToQuarter = False

    # Extract run-specific parameters
    simTag = parametrisationDict[runModel]['simTag']
    estimatesTag = parametrisationDict[runModel]['estimatesTag']
    mean = parametrisationDict[runModel]['mean']
    baseline = parametrisationDict[runModel]['baseline']
        
    if baseline is True:
        
        # ORIGINAL CALIBRATION
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
        
    else:
        
        # Load empirical samples to parameterise simulations
        fil = open('models/' + modelTag + '/' + estimatesTag + '.pkl','rb')
        datas = fil.read()
        fil.close()
        results = pickle.loads(datas,encoding="bytes")
    
        if mean is True:
        
            flat_samples = results['samples']
            params = np.mean(flat_samples, axis = 0)
    
        else:
                
            params = results['mode']
    
    # Create logging directory
    log_path = "logs//" + simTag
    print('Saving logs to: ' + log_path)
    os.makedirs(log_path,mode=0o777)

    # Create working directory
    sim_path = "simData//" + simTag
    if not os.path.exists(sim_path):
            os.makedirs(sim_path,mode=0o777)
    
    # Populate settings                
    tokens = ['P_a','P_b','P_c','P_d','P_e','P_f',
              'P_g','P_h','P_i','P_j','P_k','P_l']
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
            job_inputs.append((settings, params, numEval + i, j))

    # -- Initialise Display
    print(' ')
    print('+------------------------------------------------------+')
    print('|   Parallel Java run of SFC model - MIC measurement   |')
    print('+------------------------------------------------------+')
    print(' Model: ' + simTag )
    print(' Number of cores: ' + str(numCores) + \
        ' - Number of tasks: ' + str(numTasks))
    print('+------------------------------------------------------+')
    
    # Create pool and run parallel job
    pool = mp.Pool(processes=numCores)
    res = pool.map(runSFC,job_inputs)

    # Close pool when job is done
    pool.close()
        
    # Extract results for timer and process files
    sum_time = 0
    simDataFull = {}
    for i in range(numSamples):
        res_i = res[i]
        sum_time = sum_time + res_i
        simDataRaw = convert_sim_data(sim_path, numEval + i, numReps)
        
        # Process simulated data to observables, save
        simData = process_sim_data(simDataRaw,
                                   numBurn,
                                   numObs,
                                   monthToQuarter)
        simDataFull[i] = simData

    fil = open(sim_path + '/' + simTag + '_data.pkl','wb') 
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