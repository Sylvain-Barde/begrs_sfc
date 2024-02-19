# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 08:48:12 2021

@author: sb636
"""

import os
import sys
import pickle
import zlib
import time
import numpy as np
import torch
import gpytorch
import sampyl as smp
from tqdm import tqdm
from warnings import filterwarnings
from arviz.stats import ess
from scipy.optimize import minimize

from torch.utils.data import TensorDataset, DataLoader
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.lazy import MatmulLazyTensor, DiagLazyTensor, BlockDiagLazyTensor
from gpytorch.lazy import delazify, TriangularLazyTensor, ConstantDiagLazyTensor

#------------------------------------------------------------------------------
# Cholesky factorisation utility
def cholesky_factor(induc_induc_covar):
    
    L = psd_safe_cholesky(delazify(induc_induc_covar).double())
    
    return TriangularLazyTensor(L)

# Filter user warnings: catch CUDA initialisation warning if no GPU is present
filterwarnings("ignore", category = UserWarning)
#------------------------------------------------------------------------------
# Main classes
class begrsGPModel(gpytorch.models.ApproximateGP):
    
    # Multitask Gaussian process, with the batch dimension serving to store 
    # the multiple tasks.
    
    def __init__(self, num_vars, num_param, num_latents, num_inducing_pts):
        
        # Separate inducing points for each latent function
        inducing_points = torch.rand(num_latents, 
                                     num_inducing_pts, 
                                     num_vars + num_param)

        # Mean field variational strategy (batched) for computational speedup
        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(
            inducing_points.size(-2), 
            batch_shape = torch.Size([num_latents])
        )

        # VariationalStrategy is wrapped a LMCVariationalStrategy to combine
        # the task-level distributions into a single multivariate normal
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(self, 
                inducing_points, 
                variational_distribution, 
                learn_inducing_locations = True
            ),
            num_tasks = num_vars,
            num_latents = num_latents,
            latent_dim = -1
        )

        super().__init__(variational_strategy)

        # Mean and covariance modules 
        # Also batched, providing one module per latent variable
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents])
        )

    def forward(self, x):

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class begrs:
    
    def __init__(self, useGPU = True):
        
        # Initialise empty fields
        self.model = None
        self.likelihood = None
        self.parameter_range = None
        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY = None
        self.losses = None
        self.KLdiv = None
        
        self.N = 0
        self.num_vars = 0
        self.num_param = 0
        self.num_latents = 0
        self.num_inducing_pts = 0
        self.useGPU = useGPU
        
        if torch.cuda.is_available():
            self.haveGPU = True
        else:
            self.haveGPU = False
        
        
    def center(self, sample):
        
        mean = (self.parameter_range[:,0] + self.parameter_range[:,1])/2
        stdDev = (self.parameter_range[:,1] - self.parameter_range[:,0])/np.sqrt(12)
        
        sampleCntrd = (sample - mean)/stdDev
            
        return sampleCntrd
    
    
    def uncenter(self, sampleCntrd):
        
        mean = (self.parameter_range[:,0] + self.parameter_range[:,1])/2
        stdDev = (self.parameter_range[:,1] - self.parameter_range[:,0])/np.sqrt(12)
        
        sample = mean + stdDev*sampleCntrd
            
        return sample
    
    
    def save(self, path):
        
        print(u'\u2500' * 75)
        print(' Saving model to: {:s} '.format(path), end="", flush=True)
        
        # Check saving directory exists   
        if not os.path.exists(path):
            
            if self.model is not None:
                os.makedirs(path,mode=0o777)
                
                saveDict = {'parameter_range':self.parameter_range,
                            'trainX':self.trainX,
                            'trainY':self.trainY,
                            'testX':self.testX,
                            'testY':self.testY,
                            'losses':self.losses,
                            'KLdiv':self.KLdiv,
                            'N':self.N,
                            'num_vars':self.num_vars,
                            'num_param':self.num_param,
                            'num_latents':self.num_latents,
                            'num_inducing_pts':self.num_inducing_pts}#,
                            # 'useGPU':self.useGPU}
                
                torch.save(saveDict,
                           path + '/data_parameters.pt')
                
                torch.save(self.model.state_dict(), 
                            path + '/model_state.pt')
                torch.save(self.likelihood.state_dict(), 
                            path + '/likelihood_state.pt') 
            
                print(' - Done')
                
            else: 
                
                print('\n Cannot write to {:s}, empty model'.format(path))
            
        else:
            
            print('\n Cannot write to {:s}, folder already exists'.format(path))
            
        
    def load(self, path):
        
        print(u'\u2500' * 75)
        print(' Loading model from: {:s} '.format(path), end="", flush=True)
        
        if os.path.exists(path):
        
            parameter_path = path + '/data_parameters.pt'
            likelihood_path = path + '/likelihood_state.pt'
            model_path = path + '/model_state.pt'
        
            if os.path.exists(parameter_path):
                begrs_dict = torch.load(parameter_path)
                self.parameter_range = begrs_dict['parameter_range']
                self.trainX = begrs_dict['trainX']
                self.trainY = begrs_dict['trainY']
                self.testX = begrs_dict['testX']
                self.testY = begrs_dict['testY']
                self.N = begrs_dict['N']
                self.losses = begrs_dict['losses']
                self.KLdiv = begrs_dict['KLdiv']
                self.num_vars = begrs_dict['num_vars']
                self.num_param = begrs_dict['num_param']
                self.num_latents = begrs_dict['num_latents']
                self.num_inducing_pts = begrs_dict['num_inducing_pts']
                
            else:
                
                print("\n Missing file 'data_parameters.pt' in: {:s}".format(path))
            
            if os.path.exists(likelihood_path):
                lik_state_dict = torch.load(likelihood_path)
                self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                    num_tasks=self.num_vars)
                self.likelihood.load_state_dict(lik_state_dict, strict = False)
                
                if self.haveGPU is True and self.useGPU is True:
                    self.likelihood = self.likelihood.cuda()
            
            else: 
                
                print("\n Missing file 'likelihood_state.pt' in: {:s}".format(path))
                
            if os.path.exists(model_path):
                
                mod_state_dict = torch.load(model_path)
                self.model = begrsGPModel(self.num_vars,
                                      self.num_param,
                                      self.num_latents,
                                      self.num_inducing_pts)
                self.model.load_state_dict(mod_state_dict, strict = False)
                
                if self.haveGPU is True and self.useGPU is True:
                    self.model = self.model.cuda()
            
            else: 
                
                print("\n Missing file 'model_state.pt' in: {:s}".format(path))
            
            print(' - Done')
        
        else:
            
            print('\n Cannot load from {:s}, folder does not exist'.format(path))
        
        
    def setTrainingData(self, trainingData, doeSamples, parameter_range):
        
        print(u'\u2500' * 75)
        print(' Setting training data set', end="", flush = True)
        
        # allocate self.num_vars and self.num_param from datasets
        numSamples = doeSamples.shape[0] 
        
        if numSamples == trainingData.shape[2] and doeSamples.shape[1] == parameter_range.shape[0]:
        
            self.parameter_range = parameter_range
            self.num_param = doeSamples.shape[1] 
            numObs = trainingData.shape[0]
            self.num_vars = trainingData.shape[1]
            paramInds = self.num_vars + self.num_param
    
            samples = self.center(doeSamples)
            
            train_x_array = np.zeros([numSamples*(numObs-1),paramInds])
            train_y_array = np.zeros([numSamples*(numObs-1),self.num_vars])
            
            # Repackage training data and samples
            for i in range(numSamples):
            
                y = trainingData[1:,:,i]
                x = trainingData[0:-1,:,i]
                sample = samples[i,:]
            
                train_y_array[i*(numObs-1):(i+1)*(numObs-1),:] = y
                train_x_array[i*(numObs-1):(i+1)*(numObs-1),
                              0:self.num_vars] = x
                train_x_array[i*(numObs-1):(i+1)*(numObs-1),
                              self.num_vars:paramInds] = np.tile(sample,
                                                                  (numObs-1,1))
            # Convert to tensors and store
            self.trainX = torch.from_numpy(train_x_array).float() 
            self.trainY = torch.from_numpy(train_y_array).float() 
            
            print(' - Done', flush = True)
            print(' N' + u'\u00B0' + ' of parameters: {:>5}'.format(self.num_param))
            print(' N' + u'\u00B0' + ' of variables: {:>5}'.format(self.num_vars))
            print(' N' + u'\u00B0' + ' of parameter samples: {:>5}'.format(numSamples))
            
        else:
            
            print(' Error, inconsistent sample dimensions')
            print(' N' + u'\u00B0' +' of parameters: ' + ' '*5 +'N' + u'\u00B0' + \
                  ' of parameter samples: ')
            print(' - in range matrix: {:>5}'.format(parameter_range.shape[0]),
                  end="", flush=True)
            print(' - in DOE matrix: {:>5}'.format(doeSamples.shape[1]))
            print(' - in training data: {:>5}'.format(trainingData.shape[2]),
                  end="", flush=True)
            print(' - in DOE matrix: {:>5}'.format(doeSamples.shape[0]),
                  end="", flush=True)
    
    
    def setTestingData(self, testingData):
        
        print(u'\u2500' * 75)
        print(' Setting testing data set', end="", flush=True)
        
        # Check consistency with number of variables
        if testingData.shape[1] == self.num_vars:
        
            self.N = testingData.shape[0] - 1
            
            self.testY = torch.from_numpy(testingData[1:,:]).float() 
            
            test_x_array = np.zeros([self.N, self.num_vars+self.num_param])
            test_x_array[:,0:self.num_vars] = testingData[0:-1]
            self.testX = torch.from_numpy(test_x_array).float() 
            
            # CUDA check
            if self.haveGPU is True and self.useGPU is True:
                self.testY = self.testY.cuda()
                self.testX = self.testX.cuda()
            
            self.model.eval()
            self.likelihood.eval()
            
            print(' - Done')
            print(' Precomputing Likelihood components', end="", flush=True)
            
            # Whitening (Cholesky) on the inducing points kernel covariance
            indpts = self.model.variational_strategy.base_variational_strategy.inducing_points
            induc_induc_covar = self.model.covar_module(indpts,indpts).add_jitter()
            self.L = cholesky_factor(induc_induc_covar)

            # LMC coefficients
            lmc_coefficients = self.model.variational_strategy.lmc_coefficients.expand(*torch.Size([self.num_latents]), 
                                                                                self.model.variational_strategy.lmc_coefficients.size(-1))
            lmc_factor = MatmulLazyTensor(lmc_coefficients.unsqueeze(-1), lmc_coefficients.unsqueeze(-2))
            lmc_mod = lmc_factor.unsqueeze(-3)
            self.lmcCoeff = delazify(lmc_mod)
            
            # Diagonal predictive covariance components
            variational_inducing_covar = self.model.variational_strategy.base_variational_strategy.variational_distribution.covariance_matrix
            self.middle_diag = self.model.variational_strategy.prior_distribution.lazy_covariance_matrix.mul(-1).representation()[0]
            self.middle_diag += torch.diagonal(
                variational_inducing_covar,
                offset = 0,dim1=-2,dim2=-1
                )
            
            # Noise components
            task_noises = DiagLazyTensor(self.likelihood.noise_covar.noise)
            noise = ConstantDiagLazyTensor(
                self.likelihood.noise, 
                diag_shape=task_noises.shape[-1])
            self.noise = delazify(task_noises + noise)
            
            print(' - Done')
            
        else:
            
            print(' - Error, inconsistent number of variables')
            print(' N' + u'\u00B0' +' of variables:')
            print(' - in training data: {:>5}'.format(self.num_vars),
                  end="", flush=True)
            print(' - in test data: {:>5}'.format(testingData.shape[1]))


    def train(self, num_latents, num_inducing_pts, batchsize, epochs, 
              learning_rate = 1e-3, shuffle = True):
        
        print(u'\u2500' * 75)
        print(' Training gaussian surrogate model')
        self.num_latents = num_latents
        self.num_inducing_pts = num_inducing_pts
        
        # CUDA check
        if self.haveGPU:
            
            print(' CUDA availabe',end="", flush=True)
            
            if self.useGPU:
                
                print(' - Using GPU', flush=True)
                
            else:
                
                print(' - Using CPU', flush=True)
            
        else:
            
            print(' CUDA not availabe - Using CPU')

        
        # Create data loader from training data
        train_dataset = TensorDataset(self.trainX, self.trainY)
        train_loader = DataLoader(train_dataset, 
                                  batch_size = batchsize, 
                                  shuffle = shuffle)
    
        # Initialise model and training likelihood in training mode
        self.model = begrsGPModel(self.num_vars,
                                  self.num_param,
                                  self.num_latents,
                                  self.num_inducing_pts)
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks = self.num_vars)

        self.model.train()
        self.likelihood.train()
        
        # set Adam optimiser for the parameters
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
            ], lr=learning_rate)
        
        # ELBO loss function for model training
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, 
                                            self.model, 
                                            num_data = self.trainY.size(0))
       
        # Run optimisation
        self.losses = []
        self.KLdiv = []
        for i in range(epochs):
            
            # Within each iteration, we will go over each minibatch of data
            minibatch_iter = tqdm(train_loader, 
                                  desc='Iteration {:<3}'.format(i+1), 
                                  leave = True,
                                  file=sys.stdout)
            
            iterLoss = []
            iterKL = []
            for x_batch, y_batch in minibatch_iter:
                optimizer.zero_grad()
                output = self.model(x_batch)
                loss = - mll(output, y_batch)
                minibatch_iter.set_postfix(loss=loss.item())
                minibatch_iter.refresh()
                loss.backward()
                optimizer.step()
                
                # Save loss and KL divergence for diagnostics
                kl_term = self.model.variational_strategy.kl_divergence().div(
                    self.trainY.size(0))
                iterLoss.append(loss.item())
                iterKL.append(kl_term.item())
                
            minibatch_iter.close()
            self.losses.append(iterLoss)
            self.KLdiv.append(iterKL)


    def logP(self, theta_base, batch = False, batch_size = 40):

        # paramInds = self.num_vars + self.num_param
        
        theta = torch.from_numpy(theta_base[None,:]).float()
        theta.requires_grad_(True)
        
        # testX = self.testX.clone()
        # testX[:,self.num_vars:paramInds] = theta.repeat((self.N,1))

        if batch is False:
            
            logP = self.logPSingle(theta)
            
        else:
            
            logP = self.logPBatched(theta,batch_size)

        # logP.backward(retain_graph=True)
        theta_grad = theta.grad
            
        if self.haveGPU is True and self.useGPU is True:
            
            returnValue = (np.float64(
                                logP.detach().cpu().numpy()),
                           np.float64(
                               theta_grad.detach().cpu().numpy().flatten()))
            
        else:
                
            returnValue = (np.float64(
                                logP.detach().numpy()),
                           np.float64(
                               theta_grad.detach().numpy().flatten()))
            
        return returnValue
    
    
    def logHessian(self, theta_base, batch = False, batch_size = 40):

        # paramInds = self.num_vars + self.num_param
        
        theta = torch.from_numpy(theta_base[None,:]).float()
        theta.requires_grad_(True)
        
        # testX = self.testX.clone()
        # testX[:,self.num_vars:paramInds] = theta.repeat((self.N,1))
        
        if batch is False:
            
            # H = torch.autograd.functional.hessian(self.logPSingle,testX)
            LogP = lambda *args:  self.logPSingle(*args)

        else:
            
            # H = torch.autograd.functional.hessian(self.logPBatched,testX)
            LogP = lambda *args:  self.logPBatched(*args,batch_size)
            
        H = torch.autograd.functional.hessian(LogP, 
                                              theta,
                                              ).reshape(len(theta_base),
                                                        len(theta_base))
            
        if self.haveGPU is True and self.useGPU is True:
            
            returnValue = np.float64(H.detach().cpu().numpy())
            
        else:
                
            returnValue = np.float64(H.detach().numpy())

        return returnValue


    def logPSingle(self, theta):
        
        paramInds = self.num_vars + self.num_param
        testX = self.testX.clone()
        testX[:,self.num_vars:paramInds] = theta.repeat((self.N,1))
                          
        n = self.testY.shape[0]*self.testY.shape[1]
        batch = torch.cat([testX.unsqueeze(0)]*self.num_latents, dim=0)
        indpts = self.model.variational_strategy.base_variational_strategy.inducing_points

        
        preds = self.likelihood(self.model(batch))
        diff = self.testY.reshape([n]) - preds.mean.reshape([n])
        
        #--------------------------------
        # Uses the precomputed components set with the testing data
        # Covariance kernel components
        induc_data_covar = self.model.covar_module(indpts,batch).evaluate()
        data_data_covar = self.model.covar_module(batch,batch)
        
        # Diagonal predictive covariance
        interp_term = self.L.inv_matmul(induc_data_covar.double()).to(testX.dtype)
        predictive_covar_diag = (
            interp_term.pow(2) * self.middle_diag.unsqueeze(-1)
            ).sum(-2).squeeze(-2)
        predictive_covar_diag += torch.diagonal(
            delazify(data_data_covar.add_jitter(1e-4)),
            offset = 0,dim1=-2,dim2=-1
            )
        
        # Block diagonal LMC wrapping of diagonal predictive covariance
        covar_mat_blocks = torch.mul(self.lmcCoeff, 
                                      predictive_covar_diag.unsqueeze(-1).unsqueeze(-1)
                                      ).sum(-4) + self.noise
        covar_mat = BlockDiagLazyTensor(covar_mat_blocks)
        
        # Gaussian likelihood calculation
        inv_quad_term = covar_mat.inv_quad(diff.unsqueeze(-1))
        logP = -0.5 * sum([inv_quad_term, 
                           sum(covar_mat_blocks.det().log()),
                           diff.size(-1) * np.log(2 * np.pi)])
        
        logP.backward(retain_graph=True)
        
        return logP    
    
    def logPBatched(self, theta, batch_size):
        
        paramInds = self.num_vars + self.num_param
        testX = self.testX.clone()
        testX[:,self.num_vars:paramInds] = theta.repeat((self.N,1))
    
        # Create data loader for batches
        test_dataset = TensorDataset(testX, self.testY)
        test_loader = DataLoader(test_dataset, 
                                 batch_size=batch_size, 
                                 shuffle=False)
        
        # Get log probability for each batch directly from GPytorch
        logP = 0
        for x_batch, y_batch in test_loader:
            
            predictions = self.likelihood(self.model(x_batch))
            logP += predictions.log_prob(y_batch)
            
        logP.backward(retain_graph=True)
        
        return logP

        
    def softLogPrior(self, theta, k = 20):
        
        # convert to array, protect against overflows
        # sample = np.asarray(theta)
        sample = np.clip(np.asarray(theta),
                         -20,
                         20)
    
        f_x = np.exp(-k*(sample + 3**0.5))
        g_x = np.exp( k*(sample - 3**0.5))
    
        prior = - np.log(1 + f_x) - np.log(1 + g_x)
        grad = k*(f_x/(1+f_x) - g_x/(1+g_x))
     
        return (sum(prior),grad)
    

class begrsNutsSampler:
    """
    NUTS Sampler class for the package. See function-level help for more
    details.

    Attributes:
        
        begrsModel (begrs object)
            Instance of the begrs class
        logP (function)
            Function providing the log posterior
        mode (ndarray)
            vector of parameter values at the posterior mode
        nuts (sampyl.NUTS object)
            Instance of the sampyl NUTS sampler class
        
    Methods:
        __init__ :
            Initialises an empty instance of the class
        minESS:
            Calculates the effecgive sample size of a sample
        setup:
            Configure the NUTS sappler
        run:
            Run the NUTS sampler
    """
    
    def __init__(self,begrsModel,logP):
        """
        Initialises an empty instance of the class


        Arguments:
            begrsModel (begrs): 
                An instange of a begrs surrogate model
            logP (function): 
                A user-defined function providing the log posterior.

        """
        # Initialise empty fields
        self.begrsModel = begrsModel
        self.logP = logP
        self.mode = None
        self.nuts = None

    def minESS(self,array):
        """
        Calculates the effecgive sample size of a sample


        Arguments:
            array (ndarray):
                A posterior sample produced by the NUTS sampler

        Returns:
            float
                The effective sample size of the posterior sample.
        """
        
        essVec = np.zeros(array.shape[1])
        for i in range(array.shape[1]):
            essVec[i] = ess(array[:,i])
        
        return min(essVec)

    def setup(self, data, init):
        """
        Configure the NUTS sampler


        Arguments:
            data (ndarray):
                The empirical dataset required to calculate the likelihood.
            init (ndarray): 
                A centered vector of initial values for the parameter vector.
        """

        print('NUTS sampler initialisation')

        # Set data as the BEGRS testing set
        self.begrsModel.setTestingData(data)

        # Find posterior mode
        print('Finding MAP vector')
        t_start = time.time()
        nll = lambda *args: tuple( -i for i in self.logP(*args))
        sampleMAP = minimize(nll, init, 
                             method='L-BFGS-B',
                             bounds = self.begrsModel.num_param*[(-3**0.5,3**0.5)], 
                             jac = True)
        self.mode = sampleMAP.x
        print(' {:10.4f} secs.'.format(time.time() - t_start))
        
        # Setup NUTS 
        t_start = time.time()
        start = smp.state.State.fromfunc(self.logP)
        start.update({'sample': self.mode})
        E_cutoff = -2*self.logP(self.mode)[0]
        scale = np.ones_like(init)
        self.nuts = smp.NUTS(self.logP, 
                        start, 
                        scale = {'sample': scale},
                        grad_logp = True, 
                        step_size = 0.01, 
                        Emax = E_cutoff)

    def run(self, N, burn = 0):
        """
        Run the NUTS sampler

        Arguments:
            N (int): 
                Number of NUTS samples to draw from the posterior
            burn (int)
                Numberr of burn-in samples to discard. The default is 0.

        Returns:
        -------
            posteriorSample (ndarray): 
                (N - burn) posterior NUTS samples
        """
        
        print('NUTS sampling')
        
        # Draw samples from posterior
        t_start = time.time()
        chain = self.nuts.sample(N, burn=burn)
        
        # Process chain to uncenter samples
        posteriorSample = np.zeros([N-burn,self.begrsModel.num_param])
        for i, sampleCntrd in enumerate(chain.tolist()):
            posteriorSample[i,:] = self.begrsModel.uncenter(sampleCntrd)

        print(' {:10.4f} secs.'.format(time.time() - t_start))
        
        return posteriorSample


class begrsSbc:
    """
    Simulated Bayesian Comouting class for the package. See function-level 
    help for more details.
    
    Based on:
        
        REF

    Attributes:
        
        testSamples (ndarray)
            2-dimensional array of testing parameterizations
        testSamples (ndarray)
            3-dimensional array of simulated data
        hist (ndarray)
            Rank histogram 
        paramInd (ndarray)
            Bins for the rank histogram
        numParam (int)
            Number of parameters in a posterior sample
        numSamples (int)
            Number of samples in SBC analysis
        posteriorSampler (begrsNutsSampler)
            Instance of the begrsNutsSampler class used as the sampler
        posteriorSamplesMC (list of ndarrays)
            list of posterior samples for each testing parametrization
        posteriorSamplesESS (list of floats)
            list of effecive sample sizes for the posterior samples
            
    Methods:
        __init__ :
            Initialises an empty instance of the class
        saveData:
            Save the result of the SBC analysis
        setTestData:
            Set the  testing samples and data for the SBC analysis
        setPosteriorSampler:
            Set the posterior samplet for the SBC analysis
        run:
            Run the SBC analysis
    """
    
    def __init__(self):
        
        self.testSamples = None
        self.testData = None
        self.hist = None
        self.paramInd = None
        self.numParam = None
        self.numSamples = None
        self.posteriorSampler = None
        self.posteriorSamplesMC = []
        self.posteriorSamplesESS = []
        
    def saveData(self, path):
                
        print(u'\u2500' * 75)
        print(' Saving SBC run data to: {:s} '.format(path), end="", flush=True)
        
        # Check saving already exists
        if not os.path.exists(path):
            
            dirName,fileName = os.path.split(path)
            
            if not os.path.exists(dirName):
                os.makedirs(dirName,mode=0o777)
            
            saveDict = {'testSamples':self.testSamples,
                        'testData':self.testData,
                        'hist':self.hist,
                        'posteriorSamples':self.posteriorSamplesMC}

            fil = open(path,'wb') 
            fil.write(zlib.compress(pickle.dumps(saveDict, protocol=2)))
            fil.close()
        
            print(' - Done')
            
        else:
            
            print('\n Cannot write to {:s}, file already exists'.format(path))
    
    def setTestData(self, testSamples, testData):
        
        print(u'\u2500' * 75)
        print(' Setting test samples & data', end="", flush=True)

        if testSamples.shape[0] == testData.shape[-1]:
        
            self.numSamples = testData.shape[-1]
            self.testSamples = testSamples
            self.testData = testData
            print(' - Done')
            
        else:
            print(' - Error, inconsistent number of samples')
            print(' N' + u'\u00B0' +' of samples:')
            print(' - in test samples: {:>5}'.format(testSamples.shape[0]))
            print(' - in test data: {:>5}'.format(testData.shape[-1]))
        
    def setPosteriorSampler(self, posteriorSampler):
        
        self.posteriorSampler = posteriorSampler
        
    def run(self, N, burn, init, autoThin = True, essCutoff = 0):
        
        print(u'\u2500' * 75)
        print(' Running SBC analysis', end="", flush=True)
        
        # Consistency checks here - make sure we have everything       
        if self.testSamples is None:
            print(' - Error, no parameter samples provided')
            print(u'\u2500' * 75)
            return
        
        if self.testData is None:
            print(' - Error, no test data provided for samples')
            print(u'\u2500' * 75)
            return
        
        if self.posteriorSampler is None:
            print(' - Error, no posterior sampler provided')
            print(u'\u2500' * 75)
            return
        
        if self.hist is not None:
            print(' - Error, already run & histogram exists')
            print(u'\u2500' * 75)
            return
        
        # Run analysis if all checks passed
        L = N-burn
        for ind in range(self.testData.shape[-1]):
            
            print('Sample: {:d} of {:d}'.format(ind+1,
                                                self.numSamples))
            # Setup sampler
            data = self.testData[:,:,ind]
            testSample = self.testSamples[ind]
            
            self.posteriorSampler.setup(data, init)
            
            # Get a first sample and determine ESS
            posteriorSamples = self.posteriorSampler.run(N, burn)
            sampleESS = self.posteriorSampler.minESS(posteriorSamples)
            print('Minimal sample ESS: {:.2f}'.format(sampleESS))
            L_2 = posteriorSamples.shape[0]
            
            # If auto thin active and ESS inssufficient, retake samples
            if autoThin:
                while sampleESS < 0.95*L:
                    
                    # Adjust sample ESS if below specified cutoff
                    adjustedSampleESS = max(sampleESS,essCutoff)
                    thinRatio = L/adjustedSampleESS - 1
                    
                    addSamples = self.posteriorSampler.run(
                                        np.ceil(L_2*thinRatio).astype(int))
                    posteriorSamples = np.concatenate((posteriorSamples,
                                                      addSamples), 
                                                      axis = 0)
                    sampleESS = self.posteriorSampler.minESS(posteriorSamples)
                    print('Minimal sample ESS: {:.2f}'.format(sampleESS))
                    L_2 = posteriorSamples.shape[0]

            # Initialise histogram on first run
            if self.hist is None:
                self.numParam = len(testSample)
                self.paramInd = np.arange(self.numParam)
                self.hist = np.zeros([L + 1, self.numParam])
            
            # Augment histogram and MC collection of posterior samples
            rankStatsRaw = sum(posteriorSamples < testSample)
            rankStats = np.floor(rankStatsRaw * L/L_2).astype(int)
            self.hist[rankStats,self.paramInd]+=1
            self.posteriorSamplesMC.append(posteriorSamples)
            self.posteriorSamplesESS.append(sampleESS)
            
        print(' SBC analysis complete')
        print(u'\u2500' * 75)
