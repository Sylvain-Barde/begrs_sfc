# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 09:59:44 2018

This file contains the functions required to simulate the SFC model in 
parallel.

@author: Sylvain Barde, University of Kent
"""

import os
import time
import subprocess
import shlex
import re
import shutil
import numpy as np
import pandas as pd
import zipfile
import pickle
import zlib
import sobol

#------------------------------------------------------------------------------
def parametrise(tokens, paramVec, sampleID, rep, numReps, numSims, numObs, 
                pathIn = 'setup', pathOutRoot = 'Temp'):
    """
    Parametrises a run of the SFC model by copying configuration files into a 
    dedicated run folder, and replacing token parameters values by the entries 
    provided. The base configuration files are provided in the 'pathIn' folder.

    Arguments:
        tokens (list of strings):
            strings to be replaced by parameter values in the various SFC 
            configuration files.
        paramVec (ndarray):
            Vector of parameter values to replace
        sampleID (int):
            ID of the sample (used to generate run folder name).
        rep (int):
            parallel replication ID of the parameterisation (when the same 
            parametrisation is sent to multiple cores)
        numReps (int):
            Number of parallel replications to be carried out on a given 
            parameterisation.
        numSims (int):
            Number of simulations to run on the same parametrization (single 
            process)
        numObs (int):
            Number of time-series observations to simulate.
        pathIn (str)
            Path to locatin of base configuration files. The default is 
            'setup'.
        pathOutRoot (str):
            Path to location of parametrised configuration files. The default 
            is 'Temp'.

    """
    
    # File names and tokens
    main_in = 'main_base.xml'        # Name of main file
    model_in = 'model_base.xml'      # Name of model file

    sampleStr = str(sampleID) + '-' + str(rep)
    seed = sampleID*numReps + rep
    
    pathOut = pathOutRoot + '/sample_' + sampleStr   # Path to Models
    os.makedirs(pathOut,mode=0o777)
    
    # Copy across common files
    shutil.copyfile(pathIn + '/log4j.xml',pathOut +'/log4j.xml')
    shutil.copyfile(pathIn + '/reports.xml',pathOut +'/reports.xml')
    
    # Open base files
    mainFile = open(pathIn + '/' + main_in,'r')
    mainTextBase = mainFile.read()
    mainFile.close()
    
    modelFile = open(pathIn + '/' + model_in,'r')
    modelTextBase = modelFile.read()
    modelFile.close()
    
    taskPath = pathOutRoot + '//sample_' + sampleStr + '//'
    
    # Replace parameters with correct values in Main file
    mainText = re.sub('NUMTASK',sampleStr,mainTextBase)
    mainText = re.sub('NUMSIM',str(numSims),mainText)
    mainText = re.sub('DATA_PATH',taskPath,mainText)
    
    # Generate out name for main file and save
    fileName = re.sub('base', sampleStr, main_in)
    mainFile = open(pathOut + '/' + fileName,'w')
    mainFile.write(mainText)
    mainFile.close()
    
    # Replace parameters with correct values in Model file
    modelText = re.sub('NUMOBS',str(numObs),modelTextBase)
    modelText = re.sub('RNGSEED',str(seed),modelText)
    for i,token in enumerate(tokens):
        modelText = re.sub(token,str(paramVec[i]),modelText)
    
    # Generate out name for model file and save
    fileName = re.sub('base', sampleStr, model_in)
    modelFile = open(pathOut + '/' + fileName,'w')
    modelFile.write(modelText)
    modelFile.close()        

#------------------------------------------------------------------------------
def remove_files(path, sampleID, rep):
    """
    Clean-up utility, used to remove configuration files once a simualation 
    is finished

    Arguments:
        path (str):
            Path to location of parametrised configuration files. 
        sampleID (int):
            ID of the sample (used to generate run folder name).
        rep (int):
            parallel replication ID of the parameterisation (when the same 
            parametrisation is sent to multiple cores)
    """
    
    files = ['main_base.xml',
             'model_base.xml',
             'log4j.xml',
             'reports.xml']

    sampleStr = str(sampleID) + '-' + str(rep)
    for file in files:
        fileName = re.sub('base', sampleStr,file)
        fileNameFull = path + '/sample_' + sampleStr + '/' + fileName
        os.remove(fileNameFull)

#------------------------------------------------------------------------------
def get_sobol_samples(num_samples, parameter_support, skips):
    """
    Draw multi-dimensional sobol samples

    Arguments:
        num_samples (int):
            Number of samples to draw.
        parameter_support (ndarray):
            A 2D set of bounds for each parameter. Structure is:
                2 x numParams
            with row 0 containing lower bounds, row 1 upper bounds
                
        skips (int):
            Number of draws to skip from the start of the sobol sequence

    Returns:
        sobol_samples (ndarray):
            A 2D set of Sobol draws. Structure is:
                num_samples x num_param
    """
    params = np.transpose(parameter_support)
    sobol_samples = params[0,:] + sobol.sample(
                        dimension = parameter_support.shape[0], 
                        n_points = num_samples, 
                        skip = skips
                        )*(params[1,:]-params[0,:])
    
    return sobol_samples
#------------------------------------------------------------------------------
def convert_sim_data(folder, sampleID, numReps):
    """
    Used to process the raw simulation data into a datset of 7 observable 
    variables used in the Smets & Wouters US dataset

    Arguments:
        folder (str):
            Path to location of the raw simulation outputs.
        sampleID (int):
            ID of the sample (used to generate run folder name).
        numReps (int):
            Number of parallel replications to be carried out on a given 
            parameterisation.

    Returns:
        simData (ndarray): 
            A 3D array containing raw time-series simulations for the 7 
            observable variables of the Smets & Wouters dataset. Structure is:
                numObs x 7 x (numReps x numSims)
    """
    
    fileNames = ['unemploymentX.csv',
                 'banksLoanAvInterestX.csv',
                 'cAvPriceX.csv',
                 'nominalGDPX.csv',
                 'nominalInvestmentX.csv',
                 'hhAvWageX.csv']
    
    numVars = len(fileNames)
    
    samplePath = '/sample_' + str(sampleID)
    folderPath = folder + samplePath
    os.makedirs(folderPath,mode=0o777)
    
    # Generate zip file and delete raw csv files to save space
    z = zipfile.ZipFile(folderPath + "//simDataRaw.zip", "w",
                        zipfile.ZIP_DEFLATED)
    
    for rep in range(numReps):
    
        folderRep = folderPath + '-' + str(rep)
        fileList = list(os.walk(folderRep))[0][2]
        numSims = int(len(fileList) / numVars)
    
        sims = 0
        for sim in range(numSims):
            
            col = 0
            for fileName in fileNames:
                
                loadFile = folderRep + '/' + re.sub('X',str(sim+1),fileName)
                data = pd.read_csv(loadFile,index_col=0,header=None).values
                
                if rep == 0 and sim == 0 and col == 0:
                    N = data.shape[0]
                    simData = np.zeros([N,numVars,numSims*numReps])
                
                simData[:,col,sims+rep] = data.flatten()
                col += 1
                
            sims += numReps
        
        # Move raw csv files to zip and delete to save space
        with os.scandir(folderRep) as folderContents:
            for file in folderContents:
                        
                fileNameFull = folderRep + '//' + file.name
                fileNameZip = samplePath + '-' + str(rep) + '//' + file.name
                z.write(fileNameFull,fileNameZip)
                os.remove(fileNameFull)
                
        os.rmdir(folderRep)
        
    z.close()
    
    # -- Pickle results for future use        
    fil = open(folderPath + '//simData.pkl','wb')
    fil.write(zlib.compress(pickle.dumps(simData, protocol=2)))
    fil.close()  
    
    return simData
#------------------------------------------------------------------------------
def process_sim_data(simDataBase, burn, numObs, monthToQuarter = False):
    """
    Process the simulated data into the form required for comparability with
    Smets & Wouters, e.g deviation from trend for labour hours, log difference
    for output, consumption, etc. See Smets & Wouters 2007 for more detail

    Arguments:
        simDataBase (ndarray)
            A 3D array containing time-series simulations for the 7 observable 
            varialbes of the SMets & Wouters dataset. Structure is:
                numObs x 7 x (numReps x numSims)
        burn (int):
            Number of burn-in observatios to discard
        numObs (int):
            Desired number of observations (simulations may contain less than 
            the desired number).
        monthToQuarter (boolean): 
            Flag that raw output is in months and needs converting to quarters.
            The default is False.

    Returns:
        simDataProcessed (ndarray):
            A 3D array containing processed time-series simulations for the 7 
            observable variables of the Smets & Wouters dataset. Structure is:
            numObs x 7 x (numReps x numSims).
    """
    
    # Check raw data dimensions, pad with Nans if series is incomplete.
    # Padding will throw warnings during processing, can be ignored.
    Nsim,Nvars,reps = simDataBase.shape
    
    if Nsim < numObs:
        
        nanPadding = np.empty((numObs-Nsim,Nvars,reps))
        nanPadding[:] = np.NaN
        simDataBase = np.append(simDataBase,nanPadding,0)
    
    
    # Discard burn-in period for all reps
    simData = simDataBase[burn:Nsim,:,:]
    
    if monthToQuarter is False:
        outLen = numObs - burn - 1
    else:
        outLen = int(np.floor((numObs - burn)/3)) - 1
            
    simDataProcessed = np.zeros([outLen,7,reps])
    
    j = 0
    for rep in range(reps):
        
        simRepIn = simData[:,:,rep]
        
        # extract variables - convert to quarterly if needed
        if monthToQuarter is False:
            
            unemp = simRepIn[:,0].flatten()
            rate = simRepIn[:,1].flatten()
            cpi = simRepIn[:,2].flatten()
            gdp = simRepIn[:,3].flatten()
            inv = simRepIn[:,4].flatten()
            wage = simRepIn[:,5].flatten()
            
        else:
            
            unemp = np.zeros(outLen + 1)
            rate = np.zeros(outLen + 1)
            cpi = np.zeros(outLen + 1)
            gdp = np.zeros(outLen + 1)
            inv = np.zeros(outLen + 1)
            wage = np.zeros(outLen + 1)

            rateQuart = 1
            realGdpQuart = 0
            monthCount = 0
            
            k = 0
            for monthObs in simRepIn:
                
                if k < outLen + 1:
                    unemp[k] += monthObs[0]/3
                    rateQuart *= 1 + monthObs[1]
                    gdp[k] += monthObs[3]
                    realGdpQuart += monthObs[3]/monthObs[2]
                    inv[k] += monthObs[4]
                    wage[k] += monthObs[5]/3
                    
                monthCount += 1
                
                if monthCount == 3:
                    
                    rate[k] = rateQuart - 1
                    cpi[k] = gdp[k]/realGdpQuart

                    rateQuart = 1
                    realGdpQuart = 0
                    monthCount = 0
                    
                    k += 1
           
        # Process labour
        Lobs = 100*(1-unemp)
        Lobs = np.delete(Lobs,0)
        valid = np.logical_and(np.isinf(Lobs) == False, 
                               np.isnan(Lobs) == False)
        Lobs[np.where(np.isnan(Lobs))[0]] = 0
        Lobs[np.where(np.isinf(Lobs))[0]] = 0
        Lobs[valid] += -np.mean(Lobs[valid])
        
        # Process interest
        r = 100*rate
        r[np.where(np.isnan(r))[0]] = -5
        r[np.where(np.isinf(r))[0]] = -5
        r = np.delete(r,0)
        
        # Process cpi inflation
        cpi = 100*np.log(cpi)
        pi = np.diff(cpi)
        pi[np.where(np.isnan(pi))[0]] = -10
        pi[np.where(np.isinf(pi))[0]] = -10
        pi[np.where(np.iscomplex(pi))[0]] = -10
        
        # Process GDP
        dy = np.diff(100*np.log(gdp) - cpi)
        dy[np.where(np.isnan(dy))[0]] = -10
        dy[np.where(np.isinf(dy))[0]] = -10
        dy[np.where(np.iscomplex(dy))[0]] = -10
        
        # Process investment
        dinv = np.diff(100*np.log(inv) - cpi)
        dinv[np.where(np.isnan(dinv))[0]] = -10
        dinv[np.where(np.isinf(dinv))[0]] = -10
        dinv[np.where(np.iscomplex(dinv))[0]] = -10
        
        # Process wages
        dw = np.diff(100*np.log(wage) - cpi)
        dw[np.where(np.isnan(dw))[0]] = -10
        dw[np.where(np.isinf(dw))[0]] = -10
        dw[np.where(np.iscomplex(dw))[0]] = -10
        
        # Process consumption
        cons = gdp - inv
        dc = np.diff(100*np.log(cons) - cpi)
        dc[np.where(np.isnan(dc))[0]] = -10
        dc[np.where(np.isinf(dc))[0]] = -10
        dc[np.where(np.iscomplex(dc))[0]] = -10
        
        simRepOut = np.concatenate((Lobs[:,None],
                                    r[:,None],
                                    pi[:,None],
                                    dy[:,None],
                                    dc[:,None],
                                    dinv[:,None],
                                    dw[:,None]),
                                    axis = 1)

        simDataProcessed[:,:,j] = simRepOut
        j+=1
    
    return simDataProcessed   
#------------------------------------------------------------------------------
def runSFC(inputs):
    """
    Wrapper function for parallel run of the SFC model. Controls setup, 
    configuration, Java simulation, post simulation clean-up of confirguration 
    files and processing of simulation output.

    Arguments:
        inputs (list):
            List of configuration parameters passed to the run.

    Returns:
        toc (float):
            Simulation time in seconds
    """

    tic = time.time()
    
    # Unpack parameters
    settings = inputs[0]
    to_be_evaluated = inputs[1]
    sample = inputs[2]
    rep = inputs[3]
    
    modelTag = settings['modelTag']
    tokens = settings['tokens']
    setupPath = settings['setup_path']
    javaFiles = settings['javaFiles']
    numReps = settings['numReps']
    numObs = settings['numObs']
    numSims = settings['numSims']
    log_path = settings['log_path']
    numEval = settings['numEval']
    
    # -- Declare task initialisation
    sampleID = int(numEval + sample)
    print (' Sample {:3d}-{:d} initialised'.format(sampleID,rep))
    
    # -- Generate run files for model
    if len(to_be_evaluated.shape) == 1:
        param_vec = to_be_evaluated
    elif len(to_be_evaluated.shape) == 2:
        param_vec = to_be_evaluated[sample]
        
    parametrise(tokens, param_vec, sampleID, rep, numReps, numSims, numObs, 
                pathIn = setupPath, pathOutRoot = modelTag)
    
    # Print task/process information to log file
    sampleStr = str(sampleID) + '-' + str(rep)
    log_name = log_path + '//log_' + sampleStr + '.out'
    with open(log_name, 'w') as f_out:
        f_out.write(' Task number :   {:3d}-{:d}\n'.format(sampleID,rep))
        f_out.write(' Parent process: {:10d}\n'.format(os.getppid()))
        f_out.write(' Process id:     {:10d}\n'.format(os.getpid()))
    f_out.close
    
    # Launch Java process via a shell call
    sh_str = 'java -Djabm.config=' + \
        modelTag +'//sample_'+ sampleStr + '//main_' + sampleStr + '.xml' + \
        ' -Xmx2G -Xms1G -Dfile.encoding=Cp1252' + \
        ' -cp ' +  javaFiles[0] + ' ' + javaFiles[1]
        # ' -cp benchmark_export.jar benchmark.Main'
    args = shlex.split(sh_str)       
    
    f = open(log_name,'a')
    subprocess.run(args,stdout=f)

    # Delete run files from folder
    remove_files(modelTag, sampleID, rep)
    
    # Print completetion time
    toc = time.time() - tic
    print(' Sample number {:3d}-{:d} complete - {:10.4f} secs.'.format(
            int(sampleID), rep, toc))

    # Return output (must be pickleable)
    return (toc)
#------------------------------------------------------------------------------