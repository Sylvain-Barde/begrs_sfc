# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 09:03:46 2016

This script runs the MIC comparison of the parameters estimates obtained by 
BEGRS against the original calibration, in section 4.2 of the paper.

NOTE: While not as time-consuming as running the SFC model, the script uses the 
multiprocessing package to parallelize the MIC scoring over a large number of 
cores. Users should ensure they have access to the required compoute resources 
before running the script.

@author: Sylvain Barde, University of Kent
"""

import sys
import os
import time
import pickle
import zipfile
import zlib

import numpy as np
import multiprocessing as mp
import mic.toolbox as mt

#------------------------------------------------------------------------------
def wrapper(inputs):
    """ wrapper function"""

    tic = time.time()
    
    # Unpack inputs and parameters
    params = inputs[0]
    var_vec_base = inputs[1]
    task    = inputs[2]
    
    sim_dir    = params['sim_dir']
    emp_dir    = params['emp_dir']
    log_path   = params['log_path']
    mod_name   = params['mod_name']
    dat_name   = params['dat_name']
    lb         = params['lb']
    ub         = params['ub']
    r_vec      = params['r_vec']
    hp_bit_vec = params['hp_bit_vec']
    mem        = params['mem']
    lags       = params['lags']
    d          = params['d']
    num_runs   = params['num_runs']
    
    # -- Declare task initialisation
    print (' Task number {:3d} initialised'.format(task))
    
    # Load data
    sim_path = sim_dir + '//' + mod_name + '//' + mod_name + '_data.pkl'
    print (' Training load path:      ' + sim_path)
    fil = open(sim_path,'rb')
    datas = zlib.decompress(fil.read(),0)
    fil.close()
    simData = pickle.loads(datas,encoding="bytes")
    
    emp_path = emp_dir + '//' + dat_name
    emp_data = np.loadtxt(emp_path, delimiter="\t") 
    
    emp_data_struct = mt.bin_quant(emp_data,lb,ub,r_vec,'notests')
    emp_data_bin = emp_data_struct['binary_data']

    scores = []     # Initialise score output

    # -- iterate over conditioned variables
    for i in range(7):  
        var_vec = var_vec_base[i:7]
        var = var_vec[0]
            
        var_str = ''
        for var_i in var_vec:
            var_str = var_str + str(var_i)   
        
        # Redirect output to file and print task/process information
        file_name = 'task_' + str(task) + '_var_' +  var_str + '.out'
        sys.stdout = open(log_path + '//' + file_name, "w")
    
        print (' Task number :   {:3d}'.format(task))
        print (' Variables   :   ' + var_str)
        print (' Parent process: {:10d}'.format(os.getppid()))
        print (' Process id:     {:10d}'.format(os.getpid()))

        # -- Generate permutation from data
        perm = mt.corr_perm(emp_data, r_vec, hp_bit_vec, var_vec, lags, d)

        # - Stage 1 - train tree with training data
        tag = 'Model setting ' + mod_name
        for j in range(num_runs):
            for i in range(simData[j].shape[2]):
                dat = simData[j][:,:,i]
                data_struct = mt.bin_quant(dat,lb,ub,r_vec,'notests') # Discretise
                data_bin = data_struct['binary_data']
                print('\n Training series {:}, replication {:}'.format(j+1,i+1))
                
                if j == 0 and i == 0:
                    output = mt.train(None, data_bin, mem, lags, d, var, tag, perm)
                else:
                    T = output['T']
                    output = mt.train(T, data_bin, mem, lags, d, var,tag, perm)

        T.desc()
        
        # - Stage 2 - Score empirical series with tree
        score_struct = mt.score(T, emp_data_bin)
        mic_vec = score_struct['score'] - score_struct['bound_corr']
        scores.append(mic_vec)
        
    # Redirect output to console and print completetion time
    sys.stdout = sys.__stdout__
    
    # Generate zip file and delete temporary logs to save space
    z = zipfile.ZipFile(log_path + "//log_task_" + str(task) + ".zip", "w",
                        zipfile.ZIP_DEFLATED)
    for i in range(7):
        var_vec = var_vec_base[i:7]           
        var_str = ''
        for var_i in var_vec:
            var_str = var_str + str(var_i)   
                
        file_name = 'task_' + str(task) + '_var_' +  var_str + '.out'
        file_name_full = log_path + '//' + file_name
        
        z.write(file_name_full,file_name)
        os.remove(file_name_full)

    z.close()
    
    # Print completetion time
    toc = time.time() - tic
    print(' Task number {:3d} complete - {:10.4f} secs.'.format(int(task),toc))

    # Return output (must be pickleable)
    return (toc,scores)

#------------------------------------------------------------------------------
if __name__ == '__main__':
    
    # Load/Save directories
    sim_dir = 'simData'
    emp_dir = 'empData'
    save_dir = 'scores'
    
    # Pick dataset from list
    dataset = 1
    datasetTags = ['us',                # 0
                   'new']               # 1
    datasetFiles = ['usdata_base.txt',  # 0
                    'usdata_new.txt']   # 1 
    
    # Pick training method from list (for robustness checks)
    method = 0
    methods = ['high_L1',               # 0
               'low_L3']                # 1
    
    # List models to be compared (data-dependednt)
    datasetTag = datasetTags[dataset]
    models = ['benchmark_sfc_baseline',
              'benchmark_sfc_{:s}_mode'.format(datasetTag),
              'benchmark_sfc_{:s}_mean'.format(datasetTag)]
        
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
        
    # Parametrise methods
    if method == 0:     # High resolution, 1 memory lag
        r_vec = [6, 6, 6, 6, 6, 6, 6]
        lags = 1
    elif method == 1:   # Low resolution, 3 memory lags
        r_vec = [3, 3, 3, 3, 3, 3, 3]
        lags = 3
    
    for model in models:
            
        t_start = time.time()     
        
        # Create logging directory
        log_path = ("logs//"  + datasetTag + '//' + methods[method] +
                    '//' + model + "//train_run_" + 
                    time.strftime("%d-%b-%Y_%H-%M-%S",
                                  time.gmtime()))
        print('Saving logs to: ' + log_path)
        os.makedirs(log_path,mode=0o777)
    
        # Create saving directory    
        save_path = (save_dir + '//' + datasetTag + '//' + 
                     methods[method] + '//' + model)
        if not os.path.exists(save_path):
            os.makedirs(save_path,mode=0o777)
    
        # Set parameters    
        params = dict(sim_dir = sim_dir,
                      emp_dir = emp_dir,
                      log_path = log_path,
                      mod_name = model,
                      dat_name = datasetFiles[dataset],
                      lb =         [-7, 0,-1,-3,-3,-10,-1],
                      ub =         [ 7, 5, 3, 3, 3, 10, 2.5],
                      r_vec =      r_vec,
                      hp_bit_vec = [ 3, 3, 3, 3, 3,  3, 3],
                      mem  = 1000000,
                      d    = 28,
                      lags = lags,
                      num_runs = 1000)
        
        # Populate job
        num_tasks = len(var_vec_base)
        num_cores = num_tasks
        job_inputs = []
        for i in range(num_tasks):
            job_inputs.append((params, var_vec_base[i], i))
        
        # -- Initialise Display
        print(' ')
        print('+------------------------------------------------------+')
        print('|           Parallel CTW - Training and scoring        |')
        print('+------------------------------------------------------+')    
        print(' Number of cores: ' + str(num_cores) + 
              ' - Number of tasks: ' + str(num_tasks))
        print('+------------------------------------------------------+')
        
        # Create pool and run parallel job
        pool = mp.Pool(processes=num_cores)
        res = pool.map(wrapper,job_inputs)
    
        # Close pool when job is done
        pool.close()
     
        # Extract results and get timer
        sum_time = 0
        for(i, var_vec) in enumerate(var_vec_base):
            res_i = res[i]
            sum_time = sum_time + res_i[0]
    
            # -- Save results (name depends on variables)
            var_str = ''
            for var_i in var_vec:
                var_str = var_str + str(var_i)   
                
            fil = open(save_path + '//scores_var_' + var_str + '.pkl','wb')
            fil.write(pickle.dumps(res_i, protocol=2))
            fil.close()   
       
        # Print timer diagnostics
        print('+------------------------------------------------------+')
        timer_1 = time.time() - t_start
        print(' Total running time:     {:10.4f} secs.'.format(timer_1))
        print(' Sum of iteration times: {:10.4f} secs.'.format(sum_time))
        print(' Mean iteration time:    {:10.4f} secs.'.format(
                sum_time/num_tasks))
        print(' Speed up:               {:10.4f}'.format(sum_time/timer_1))
        print('+------------------------------------------------------+')

#------------------------------------------------------------------------------
