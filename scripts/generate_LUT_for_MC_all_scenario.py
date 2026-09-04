#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 14:17:30 2026

@author: Ihasustaa
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

#%% get data (soil, climate, carbon inputs...) from long term experiment site QualiAgro 

data = pd.read_csv("../data/Qualiagro_input_AMG.csv")
data_LEG = data[data['ID_Treatment']=='QUA_LEG_TEM']
data_PRO = data[data['ID_Treatment']=='QUA_PRO_TEM']

t = data_LEG['Year'].size

def get_mean_t(data,r=0.65):
    mean_t={}
    mean_t['T'] =  data['Temperature'].values
    mean_t['H'] =   data['P-ETP'].values
    mean_t['Clay'] =  data['Clay'].values
    mean_t['CaCO3'] = data['CaCO3'].values
    mean_t['pH'] =   data['pH'].values
    mean_t['C/N'] =  data['C/N'].values
    mean_t['I'] =   data['Cinp_hum'].values
    mean_t['QC0'] =  data['SOC_stock'].values
    mean_t['r'] = r*np.ones(t)
    return mean_t


mean_t_LEG=get_mean_t(data_LEG)
mean_t_PRO=get_mean_t(data_PRO)

#%% AMGv2 param

param={}
param['a_calcium'] = 0.0015
param['a_pH']      = 0.112
param['b_pH']      = 8.5
param['a_clay']    = 2.519*0.001
param['a_CN']      = 0.06
param['b_CN']      = 11
param['c_CN']      = 0.8
param['d_CN']      = 0.2
param['a_H']      = 0.03
param['b_H']      = 5.247
param['a_T']      = 25
param['T_ref']    = 15
param['c_T']      = 0.12
param['b_T']      = (param['a_T']-1)*np.exp(param['c_T'] *param['T_ref'] )
param['k0']       =0.290

CV_n_LEG = {}
CV_n_PRO = {}

#%% 
## relative error dictionnary


def err_rel_dict(err_r, err_I, err_QC0, err_T, err_else) :
    err_rel={}
    keys_else = [ 'H', 'Clay', 'pH', 'C/N']
    for k in keys_else :
        err_rel[k] = err_else
    err_rel['r'] = err_r
    err_rel['I'] = err_I
    err_rel['QC0'] = err_QC0
    err_rel['T'] = err_T
    return err_rel
    

### Relative error for carbon inputs depending on error scenario

### NO straw export 
#err_rel_I_DAM = 0.21 # in % : found in Cinp study case using DAM and DBM with Pse=1 => no straw export
#err_rel_I_Yp = 0.54  # in % : found in Cinp study case using Yp with Pse=1 => no straw export

### Straw export
err_rel_I_DAM = 0.30 # in % : found in Cinp study case using DAM and DBM with Pse=0.5 and sigma_Pse = 0.24
err_rel_I_Yp = 0.57  # in % : found in Cinp study case using Yp with Pse=0.5 and sigma_Pse = 0.24


### Relative error for pedoclimatic variables 

err_rel_T = 0.01  # in % : found comparing local and ERA5 T in Vanella et al. 2022
err_rel_10 = 0.1  # in % :to represent typical relative error of local measurement
err_rel_QC0 = 0.1 # in % :to represent typical relative error of local measurement



## Here, the error in initial SOC stock QC0 comes from the spatial variability of the stock SOC in the plot treatment,
## based on 4 replicats.
err_rel_QC0_LEG = np.sqrt(data_LEG['Variance_SOC_stock'].iloc[0])/mean_t_LEG['QC0'][0]
err_rel_QC0_PRO = np.sqrt(data_PRO['Variance_SOC_stock'].iloc[0])/mean_t_PRO['QC0'][0]


## Error on initial stable/total SOC ratio r (absolute error, not relative)

# Default initialisation scenario :
err_r_18 = 0.18 # in Clivot et al. (2019), r optimized = 60+-18%

# Initialisation using Party_SOC model with Rock-Eval analysis :
err_r_6 = 0.06  # RMSE = 0.06 in Kanari et al. (2022)




## Large-scale error scenario : 
## Constant error representing large scale products (ERA5 and LUCAS soil maps)

# sigma values for soil from Ballabio et al. (2016) and (2019) (LUCAS mapping)
sigma_LS={} #LS for large scale error scenario
sigma_LS['Clay'] = 77
sigma_LS['CaCO3']= 78.29
sigma_LS['pH'] = 0.78 # took the max sigma which correspond to pH in H20 (otherwise sigma = 0.68 for pH in CaCl2)
sigma_LS['C/N']=1.97


# sigma values for temperature, rain and PET from comparasion studies between ERA5 reanalyses and local data

sigma_LS['T']= 1.97/np.sqrt(365) # worst case in Vanella et al. (2022) : error of Tair daily from ERA5 compared with measure=> we divide by sqrt(N days) to have mean annual
sigma_LS['Rain']= 95.5 #worst case in Lavers et al. (2022)
sigma_LS['PET']= 16.81 #worst case in Vanella et al. (2022)
sigma_LS['H'] = np.sqrt(sigma_LS['Rain']**2 + sigma_LS['PET']**2)



#%% Function to create sigma vectors with time dimension t, stored in a dictionary 

def input_err_rel(mean_t, err_rel):
    sigma_t = {}
    keys = [ 'T', 'H', 'Clay', 'pH', 'C/N', 'I', 'QC0']
    for k in keys : 
        sigma_t[k] = abs(err_rel[k]*mean_t[k])
    sigma_t['CaCO3'] =  abs(np.random.normal(0,2.5,t)) #in Qualiagro, CaCO3=0, so we create a small random error
    sigma_t['r'] =  err_rel['r']*np.ones(t) 
    return sigma_t


def input_err_LS(mean_t, sigma_LS, err_rel): #LS for large scale error scenario
    sigma_t = {}
    keys = [ 'T', 'H', 'Clay', 'CaCO3', 'pH', 'C/N']
    for k in keys :
        sigma_t[k]=sigma_LS[k]*np.ones(t) 

    sigma_t['QC0'] = err_rel['QC0'] * mean_t['QC0']
    sigma_t['I'] =  err_rel['I']*mean_t['I']
    sigma_t['r'] =  err_rel['r']*np.ones(t) 
    return sigma_t


#%% Numerical sampling approach

# generate the n sampling based on mean and sigma vectors with time dimension t
def generate_LUT(mean_t, sigma_t, keys, t, n):
    LUT = {}
    for k in keys:
        LUT[k] = np.random.normal(
            mean_t[k][:, None],
            sigma_t[k][:, None],
            size=(t, n)
        )
    return LUT


#%% k mineralisation from AMGv2
def mineralisation_process(LUT):
    stress={}
   
    
    # f(A) in Clivot et al. (2019)
    stress['clay_stress'] = np.exp(- param['a_clay'] * LUT['Clay'])
    
    # f(CaCo3) in Clivot et al. (2019)
    stress['calcium_stress'] = 1/(1+  param['a_calcium']*LUT['CaCO3'])
    
    # f(pH) in Clivot et al. (2019)
    stress['pH_stress'] = np.exp(-param['a_pH'] * (LUT['pH']-param['b_pH'])*(LUT['pH']-param['b_pH']))
    
    # f(C/N) in Clivot et al. (2019)
    u = (LUT['C/N']-param['b_CN'])
    stress['CN_stress'] = 0.8*np.exp(-param['a_CN']*u*u)+0.2
    
    #f(T) in Clivot et al. (2019), here all T>0
    stress['T_stress'] = param['a_T']/((1+param['b_T']*np.exp(-param['c_T']*LUT['T'])))
    
    #f(H) in Clivot et al. (2019)
    stress['H_stress'] = 1/(1+param['a_H']*np.exp((-param['b_H']*LUT['H']/1000)))
    
    # minéralisation rate k
    stress['mineralisation_rate'] = (param['k0']*stress['CN_stress']* 
                                       stress['calcium_stress'] * stress['pH_stress'] * stress['clay_stress'] ) *                                                        stress['H_stress'] * stress['T_stress'] 
    return stress


#%% AMG 

## Function applying the AMG model on the arrays (LUT) with (t,n) dimensions,
## generated with a numerical sampling :

def AMG(LUT,t,n):
    # Compute all the stress function and the mineralisation rate k, 
    # via the mineralisation_process() function
    stress = mineralisation_process(LUT)
    
    # Initialise the var dictionnary that will store the value of the carbon pools and total SOC stock
    var ={}
    var['QCa'] = np.zeros((t,n)) # active carbon pool
    var['QC'] = np.zeros((t,n))  # stable carbon pool
    var['QCs'] = np.zeros((t,n)) # total SOC stock

    var['QCa'][0,:] = LUT['QC0'][0,:] * (1-LUT['r'][0,:]) # we initialise all the active carbon pool for all sampling
    stable_C_pool = LUT['QC0'][0, :] * LUT['r'][0, :]
    var['QCs'] = np.tile(stable_C_pool, (t, 1))    # the value of the stable C pool QCs is set at all times with its initial value as it is constant during the simu

    var['QC'][0,:] = var['QCa'][0,:]  + var['QCs'][0,:] # the total SOC stock QC is the sum of the two C pools at all time

    C_inp_hum = LUT['I'] # carbon inputs humified
    
    for i in range(t-1) : 
        dQCa = C_inp_hum[i,:] - stress['mineralisation_rate'][i,:]*var['QCa'][i,:]
        var['QCa'][i+1,:] = dQCa + var['QCa'][i,:]
        var['QC'][i+1,:] = var['QCa'][i+1,:] + var['QCs'][i+1,:]
    return var
        


#%% Main function that call all functions

### Create the sigma vectors according to the error scenario
### Generate the numerical sampling based on the sigma vectors
### Apply the AMG model on the arrays or LUT generated with the numerical sampling
### It returns two arrays .npy :  
    ### LUT : contain the input variables generated with the numerical sampling
    ### AMG_LUT : contain the output variables, i.e., SOC variables computed within AMG with LUT as inputs data
def run_experiment(
    mean_t,
    error_k_model,     # "relative" or "LS"
    err_rel,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    name_conf_err
    
    
):

    
    # 1. Build sigma_t
    if error_k_model == "relative":
        sigma_t = input_err_rel(mean_t, err_rel)
        np.save(dir_path_sigma_t/name_conf_err+'.npy', sigma_t)
    elif error_k_model == "LS":
        sigma_t = input_err_LS(mean_t, sigma_LS, err_rel)
        np.save(dir_path_sigma_t/name_conf_err+'.npy', sigma_t)
    else:
        raise ValueError("Unknown error model")

    # 2. Generate LUT
    LUT = generate_LUT(mean_t, sigma_t, keys, t, n)

    # 3. Run AMG
    AMG_LUT = AMG(LUT,t,n)

    return LUT, AMG_LUT

#%% Application 

keys = ['T', 'H', 'Clay', 'CaCO3', 'pH', 'C/N', 'I', 'QC0', 'r']

### Size of sampling n      
n = 100000 # 10⁵ kept for the results in paper, can be modified to test the effect of sampling size

### directories to save the .npy :
    
## To store the outputs of the Monte Carlo simulations
dir_path = "../outputs/LUT_MC/absolute_SOC/"
# Create the repertory if needed
dir_path = Path(dir_path)
dir_path.mkdir(parents=True, exist_ok=True)


## To store the sigma's vector that will be used for the analytical uncertainty propagation methods as inputs
dir_path_sigma_t = "../inputs/sigma_t/absolute_SOC/"
# Create the repertory if needed
dir_path_sigma_t = Path(dir_path_sigma_t)
dir_path_sigma_t.mkdir(parents=True, exist_ok=True)

"""
The name of the experiment corresponds to the different error scenarios used for the inputs variables, following this convention : 
- term1_term2_LUT_treatment for the different input arrays called LUT, with dimension (t,n) i.e., (time, sample size)
- term1_term2_AMG_LUT_treament for the different arrays containing AMG outputs called AMG_LUT, with dimension (t,n) i.e., (time, sample size)

(i) The first term 'term1' corresponds to the error scenarios concerning pedoclimatic input variables and input variables to compute intermediate variables I and k (carbon inputs and mineralisation rate).

- opti : when both scenario are Field-scale error scenario
- pessim : when both scenario are Large-scale error scenario
- ACEOkLS : when the scenario concerning carbon inputs uncertainty is Field-scale scenario ('ACEO' for using AgriCarbon-EO error), and the error scenario for computing the mineralisation rate uncertainty is Large-scale scenario (kLS for Large-Scale error).
- regstatk10 : when the scenario concerning carbon inputs uncertainty is Large-scale scenario ('regstat' for using regional statistics error), and the error scenario for computing the mineralisation rate uncertainty is Field-scale scenario (k10 as in the field-scale scenario we assume a 10% relative error).

(ii) The second term 'term2' corresponds to the error scenarios concerning error in initial conditions QC0 (initial SOC_stock), and r (initial stable/total SOC ratio).

- local : when error on initial SOC stock QCO is 10% of relative error, and the initial stable/total SOC ratio r is initialised with Party_SOC model and Rock-Eval analysis (sigma_r = 0.06)
- large : when error on initial SOC stock QCO is 20% of relative error, and the initial stable/total SOC ratio r is initialised with default value (sigma_r = 0.18)
- mesdef : when error on initial SOC stock QCO corresponds to variance of measurement in QualiAgro dataset, and the initial stable/total SOC ratio r is initialised with default values (sigma_r=0.18)
- mesRE :  when error on initial SOC stock QCO corresponds to variance of measurement in QualiAgro dataset, and the initial stable/total SOC ratio r is initialised with Party_SOC model and Rock-Eval analysis (sigma_r = 0.06)
- localdef : when error on initial SOC stock QCO is 10% of relative error, and the initial stable/total SOC ratio r is initialised with default value (sigma_r = 0.18)
- largeRE :  when error on initial SOC stock QCO is 20% of relative error, and the initial stable/total SOC ratio r is initialised with Party_SOC model and Rock-Eval analysis (sigma_r = 0.06)
"""




err_opti_local = err_rel_dict(err_r=0.06, err_I=err_rel_I_DAM, err_QC0=0.1, err_T=0.01, err_else=0.1)

opti_local_LUT_LEG, opti_local_AMG_LUT_LEG = run_experiment(
    mean_t_LEG,
    'relative',     # "relative" or "LS"
    err_opti_local,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'LEG_err_opti_local')

np.save(dir_path + "opti_local_LUT_LEG_n_{n}.npy", opti_local_LUT_LEG)
np.save(dir_path +"AMG_LUT" /f"opti_local_AMG_LUT_LEG_n_{n}.npy", opti_local_AMG_LUT_LEG)


opti_local_LUT_PRO, opti_local_AMG_LUT_PRO = run_experiment(
    mean_t_PRO,
    'relative',     # "relative" or "LS"
    err_opti_local,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'PRO_err_opti_local')

np.save(dir_path + "opti_local_LUT_PRO_n_{n}.npy", opti_local_LUT_PRO)
np.save(dir_path +"AMG_LUT" /f"opti_local_AMG_LUT_PRO_n_{n}.npy", opti_local_AMG_LUT_PRO)

err_opti_localdef = err_rel_dict(err_r=0.18, err_I=err_rel_I_DAM, err_QC0=0.1, err_T=0.01, err_else=0.1)

opti_localdef_LUT_LEG, opti_localdef_AMG_LUT_LEG = run_experiment(
    mean_t_LEG,
    'relative',     # "relative" or "LS"
    err_opti_localdef,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'LEG_err_opti_localdef')

np.save(dir_path + "opti_localdef_LUT_LEG_n_{n}.npy", opti_localdef_LUT_LEG)
np.save(dir_path +"AMG_LUT" /f"opti_localdef_AMG_LUT_LEG_n_{n}.npy", opti_localdef_AMG_LUT_LEG)


opti_localdef_LUT_PRO, opti_localdef_AMG_LUT_PRO = run_experiment(
    mean_t_PRO,
    'relative',     # "relative" or "LS"
    err_opti_localdef,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'PRO_err_opti_localdef')

np.save(dir_path + "opti_localdef_LUT_PRO_n_{n}.npy", opti_localdef_LUT_PRO)
np.save(dir_path +"AMG_LUT" /f"opti_localdef_AMG_LUT_PRO_n_{n}.npy", opti_localdef_AMG_LUT_PRO)




err_opti_large = err_rel_dict(err_r=0.18, err_I=err_rel_I_DAM, err_QC0=0.2, err_T=0.01, err_else=0.1)

opti_large_LUT_LEG, opti_large_AMG_LUT_LEG = run_experiment(
    mean_t_LEG,
    'relative',     # "relative" or "LS"
    err_opti_large,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'LEG_err_opti_large')

np.save(dir_path + "opti_large_LUT_LEG_n_{n}.npy", opti_large_LUT_LEG)
np.save(dir_path +"AMG_LUT" /f"opti_large_AMG_LUT_LEG_n_{n}.npy", opti_large_AMG_LUT_LEG)


opti_large_LUT_PRO, opti_large_AMG_LUT_PRO = run_experiment(
    mean_t_PRO,
    'relative',     # "relative" or "LS"
    err_opti_large,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'PRO_err_opti_large')

np.save(dir_path + "opti_large_LUT_PRO_n_{n}.npy", opti_large_LUT_PRO)
np.save(dir_path +"AMG_LUT" /f"opti_large_AMG_LUT_PRO_n_{n}.npy", opti_large_AMG_LUT_PRO)


err_opti_largeRE = err_rel_dict(err_r=0.06, err_I=err_rel_I_DAM, err_QC0=0.2, err_T=0.01, err_else=0.1)

opti_largeRE_LUT_LEG, opti_largeRE_AMG_LUT_LEG = run_experiment(
    mean_t_LEG,
    'relative',     # "relative" or "LS"
    err_opti_largeRE,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'LEG_err_opti_largeRE')

np.save(dir_path + "opti_largeRE_LUT_LEG_n_{n}.npy", opti_largeRE_LUT_LEG)
np.save(dir_path +"AMG_LUT" /f"opti_largeRE_AMG_LUT_LEG_n_{n}.npy", opti_largeRE_AMG_LUT_LEG)


opti_largeRE_LUT_PRO, opti_largeRE_AMG_LUT_PRO = run_experiment(
    mean_t_PRO,
    'relative',     # "relative" or "LS"
    err_opti_largeRE,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'PRO_err_opti_largeRE')

np.save(dir_path + "opti_largeRE_LUT_PRO_n_{n}.npy", opti_largeRE_LUT_PRO)
np.save(dir_path +"AMG_LUT" /f"opti_largeRE_AMG_LUT_PRO_n_{n}.npy", opti_largeRE_AMG_LUT_PRO)


## as we will use LS error, we don't care about the value of err_else et err_T

err_pessim_large = err_rel_dict(err_r=0.18, err_I=err_rel_I_Yp, err_QC0=0.2, err_T=0.01, err_else=0.1)

pessim_large_LUT_LEG, pessim_large_AMG_LUT_LEG = run_experiment(
    mean_t_LEG,
    'LS',     # "relative" or "LS"
    err_pessim_large,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'LEG_err_pessim_large')

np.save(dir_path + "pessim_large_LUT_LEG_n_{n}.npy", pessim_large_LUT_LEG)
np.save(dir_path +"AMG_LUT" /f"pessim_large_AMG_LUT_LEG_n_{n}.npy", pessim_large_AMG_LUT_LEG)


pessim_large_LUT_PRO, pessim_large_AMG_LUT_PRO = run_experiment(
    mean_t_PRO,
    'LS',     # "relative" or "LS"
    err_pessim_large,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'PRO_err_pessim_large')

np.save(dir_path + "pessim_large_LUT_PRO_n_{n}.npy", pessim_large_LUT_PRO)
np.save(dir_path +"AMG_LUT" /f"pessim_large_AMG_LUT_PRO_n_{n}.npy", pessim_large_AMG_LUT_PRO)


## as we will use LS error, we don't care about the value of err_else and err_T

err_pessim_largeRE = err_rel_dict(err_r=0.06, err_I=err_rel_I_Yp, err_QC0=0.2, err_T=0.01, err_else=0.1)

pessim_largeRE_LUT_LEG, pessim_largeRE_AMG_LUT_LEG = run_experiment(
    mean_t_LEG,
    'LS',     # "relative" or "LS"
    err_pessim_largeRE,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'LEG_err_pessim_largeRE')

np.save(dir_path + "pessim_largeRE_LUT_LEG_n_{n}.npy", pessim_largeRE_LUT_LEG)
np.save(dir_path +"AMG_LUT" /f"pessim_largeRE_AMG_LUT_LEG_n_{n}.npy", pessim_largeRE_AMG_LUT_LEG)


pessim_largeRE_LUT_PRO, pessim_largeRE_AMG_LUT_PRO = run_experiment(
    mean_t_PRO,
    'LS',     # "relative" or "LS"
    err_pessim_largeRE,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'PRO_err_pessim_largeRE')

np.save(dir_path + "pessim_largeRE_LUT_PRO_n_{n}.npy", pessim_largeRE_LUT_PRO)
np.save(dir_path +"AMG_LUT" /f"pessim_largeRE_AMG_LUT_PRO_n_{n}.npy", pessim_largeRE_AMG_LUT_PRO)


err_pessim_local = err_rel_dict(err_r=0.06, err_I=err_rel_I_Yp, err_QC0=0.1, err_T=0.01, err_else=0.1)

pessim_local_LUT_LEG, pessim_local_AMG_LUT_LEG = run_experiment(
    mean_t_LEG,
    'LS',     # "relative" or "LS"
    err_pessim_local,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'LEG_err_pessim_local')

np.save(dir_path + "pessim_local_LUT_LEG_n_{n}.npy", pessim_local_LUT_LEG)
np.save(dir_path +"AMG_LUT" /f"pessim_local_AMG_LUT_LEG_n_{n}.npy", pessim_local_AMG_LUT_LEG)


pessim_local_LUT_PRO, pessim_local_AMG_LUT_PRO = run_experiment(
    mean_t_PRO,
    'LS',     # "relative" or "LS"
    err_pessim_local,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'PRO_err_pessim_local')

np.save(dir_path + "pessim_local_LUT_PRO_n_{n}.npy", pessim_local_LUT_PRO)
np.save(dir_path +"AMG_LUT" /f"pessim_local_AMG_LUT_PRO_n_{n}.npy", pessim_local_AMG_LUT_PRO)


err_pessim_localdef = err_rel_dict(err_r=0.18, err_I=err_rel_I_Yp, err_QC0=0.1, err_T=0.01, err_else=0.1)

pessim_localdef_LUT_LEG, pessim_localdef_AMG_LUT_LEG = run_experiment(
    mean_t_LEG,
    'LS',     # "relative" or "LS"
    err_pessim_localdef,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'LEG_err_pessim_localdef')

np.save(dir_path + "pessim_localdef_LUT_LEG_n_{n}.npy", pessim_localdef_LUT_LEG)
np.save(dir_path +"AMG_LUT" /f"pessim_localdef_AMG_LUT_LEG_n_{n}.npy", pessim_localdef_AMG_LUT_LEG)


pessim_localdef_LUT_PRO, pessim_localdef_AMG_LUT_PRO = run_experiment(
    mean_t_PRO,
    'LS',     # "relative" or "LS"
    err_pessim_localdef,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'PRO_err_pessim_localdef')

np.save(dir_path + "pessim_localdef_LUT_PRO_n_{n}.npy", pessim_localdef_LUT_PRO)
np.save(dir_path +"AMG_LUT" /f"pessim_localdef_AMG_LUT_PRO_n_{n}.npy", pessim_localdef_AMG_LUT_PRO)


err_ACEOkLS_localdef = err_rel_dict(err_r=0.18, err_I=err_rel_I_DAM, err_QC0=0.1, err_T=0.01, err_else=0.1)

ACEOkLS_localdef_LUT_LEG, ACEOkLS_localdef_AMG_LUT_LEG = run_experiment(
    mean_t_LEG,
    'LS',     # "relative" or "LS"
    err_ACEOkLS_localdef,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'LEG_err_ACEOkLS_localdef')

np.save(dir_path + "ACEOkLS_localdef_LUT_LEG_n_{n}.npy", ACEOkLS_localdef_LUT_LEG)
np.save(dir_path +"AMG_LUT" /f"ACEOkLS_localdef_AMG_LUT_LEG_n_{n}.npy", ACEOkLS_localdef_AMG_LUT_LEG)


ACEOkLS_localdef_LUT_PRO, ACEOkLS_localdef_AMG_LUT_PRO = run_experiment(
    mean_t_PRO,
    'LS',     # "relative" or "LS"
    err_ACEOkLS_localdef,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'PRO_err_ACEOkLS_localdef')

np.save(dir_path + "ACEOkLS_localdef_LUT_PRO_n_{n}.npy", ACEOkLS_localdef_LUT_PRO)
np.save(dir_path +"AMG_LUT" /f"ACEOkLS_localdef_AMG_LUT_PRO_n_{n}.npy", ACEOkLS_localdef_AMG_LUT_PRO)


err_ACEOkLS_local = err_rel_dict(err_r=0.06, err_I=err_rel_I_DAM, err_QC0=0.1, err_T=0.01, err_else=0.1)

ACEOkLS_local_LUT_LEG, ACEOkLS_local_AMG_LUT_LEG = run_experiment(
    mean_t_LEG,
    'LS',     # "relative" or "LS"
    err_ACEOkLS_local,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'LEG_err_ACEOkLS_local')

np.save(dir_path + "ACEOkLS_local_LUT_LEG_n_{n}.npy", ACEOkLS_local_LUT_LEG)
np.save(dir_path +"AMG_LUT" /f"ACEOkLS_local_AMG_LUT_LEG_n_{n}.npy", ACEOkLS_local_AMG_LUT_LEG)


ACEOkLS_local_LUT_PRO, ACEOkLS_local_AMG_LUT_PRO = run_experiment(
    mean_t_PRO,
    'LS',     # "relative" or "LS"
    err_ACEOkLS_local,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'PRO_err_ACEOkLS_local')

np.save(dir_path + "ACEOkLS_local_LUT_PRO_n_{n}.npy", ACEOkLS_local_LUT_PRO)
np.save(dir_path +"AMG_LUT" /f"ACEOkLS_local_AMG_LUT_PRO_n_{n}.npy", ACEOkLS_local_AMG_LUT_PRO)


err_ACEOkLS_large = err_rel_dict(err_r=0.18, err_I=err_rel_I_DAM, err_QC0=0.2, err_T=0.01, err_else=0.1)

ACEOkLS_large_LUT_LEG, ACEOkLS_large_AMG_LUT_LEG = run_experiment(
    mean_t_LEG,
    'LS',     # "relative" or "LS"
    err_ACEOkLS_large,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'LEG_err_ACEOkLS_large')

np.save(dir_path + "ACEOkLS_large_LUT_LEG_n_{n}.npy", ACEOkLS_large_LUT_LEG)
np.save(dir_path +"AMG_LUT" /f"ACEOkLS_large_AMG_LUT_LEG_n_{n}.npy", ACEOkLS_large_AMG_LUT_LEG)


ACEOkLS_large_LUT_PRO, ACEOkLS_large_AMG_LUT_PRO = run_experiment(
    mean_t_PRO,
    'LS',     # "relative" or "LS"
    err_ACEOkLS_large,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'PRO_err_ACEOkLS_large')

np.save(dir_path + "ACEOkLS_large_LUT_PRO_n_{n}.npy", ACEOkLS_large_LUT_PRO)
np.save(dir_path +"AMG_LUT" /f"ACEOkLS_large_AMG_LUT_PRO_n_{n}.npy", ACEOkLS_large_AMG_LUT_PRO)


err_ACEOkLS_largeRE = err_rel_dict(err_r=0.06, err_I=err_rel_I_DAM, err_QC0=0.2, err_T=0.01, err_else=0.1)

ACEOkLS_largeRE_LUT_LEG, ACEOkLS_largeRE_AMG_LUT_LEG = run_experiment(
    mean_t_LEG,
    'LS',     # "relative" or "LS"
    err_ACEOkLS_largeRE,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'LEG_err_ACEOkLS_largeRE')

np.save(dir_path + "ACEOkLS_largeRE_LUT_LEG_n_{n}.npy", ACEOkLS_largeRE_LUT_LEG)
np.save(dir_path +"AMG_LUT" /f"ACEOkLS_largeRE_AMG_LUT_LEG_n_{n}.npy", ACEOkLS_largeRE_AMG_LUT_LEG)


ACEOkLS_largeRE_LUT_PRO, ACEOkLS_largeRE_AMG_LUT_PRO = run_experiment(
    mean_t_PRO,
    'LS',     # "relative" or "LS"
    err_ACEOkLS_largeRE,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'PRO_err_ACEOkLS_largeRE')

np.save(dir_path + "ACEOkLS_largeRE_LUT_PRO_n_{n}.npy", ACEOkLS_largeRE_LUT_PRO)
np.save(dir_path +"AMG_LUT" /f"ACEOkLS_largeRE_AMG_LUT_PRO_n_{n}.npy", ACEOkLS_largeRE_AMG_LUT_PRO)


err_regstatk10_localdef = err_rel_dict(err_r=0.18, err_I=err_rel_I_Yp, err_QC0=0.1, err_T=0.01, err_else=0.1)

regstatk10_localdef_LUT_LEG, regstatk10_localdef_AMG_LUT_LEG = run_experiment(
    mean_t_LEG,
    'relative',     # "relative" or "LS"
    err_regstatk10_localdef,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'LEG_err_regstatk10_localdef')

np.save(dir_path + "regstatk10_localdef_LUT_LEG_n_{n}.npy", regstatk10_localdef_LUT_LEG)
np.save(dir_path +"AMG_LUT" /f"regstatk10_localdef_AMG_LUT_LEG_n_{n}.npy", regstatk10_localdef_AMG_LUT_LEG)


regstatk10_localdef_LUT_PRO, regstatk10_localdef_AMG_LUT_PRO = run_experiment(
    mean_t_PRO,
    'relative',     # "relative" or "LS"
    err_regstatk10_localdef,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'PRO_err_regstatk10_localdef')

np.save(dir_path + "regstatk10_localdef_LUT_PRO_n_{n}.npy", regstatk10_localdef_LUT_PRO)
np.save(dir_path +"AMG_LUT" /f"regstatk10_localdef_AMG_LUT_PRO_n_{n}.npy", regstatk10_localdef_AMG_LUT_PRO)


err_regstatk10_local = err_rel_dict(err_r=0.06, err_I=err_rel_I_Yp, err_QC0=0.1, err_T=0.01, err_else=0.1)

regstatk10_local_LUT_LEG, regstatk10_local_AMG_LUT_LEG = run_experiment(
    mean_t_LEG,
    'relative',     # "relative" or "LS"
    err_regstatk10_local,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'LEG_err_regstatk10_local')

np.save(dir_path + "regstatk10_local_LUT_LEG_n_{n}.npy", regstatk10_local_LUT_LEG)
np.save(dir_path +"AMG_LUT" /f"regstatk10_local_AMG_LUT_LEG_n_{n}.npy", regstatk10_local_AMG_LUT_LEG)


regstatk10_local_LUT_PRO, regstatk10_local_AMG_LUT_PRO = run_experiment(
    mean_t_PRO,
    'relative',     # "relative" or "LS"
    err_regstatk10_local,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'PRO_err_regstatk10_local')

np.save(dir_path + "regstatk10_local_LUT_PRO_n_{n}.npy", regstatk10_local_LUT_PRO)
np.save(dir_path +"AMG_LUT" /f"regstatk10_local_AMG_LUT_PRO_n_{n}.npy", regstatk10_local_AMG_LUT_PRO)


err_regstatk10_large = err_rel_dict(err_r=0.18, err_I=err_rel_I_Yp, err_QC0=0.2, err_T=0.01, err_else=0.1)

regstatk10_large_LUT_LEG, regstatk10_large_AMG_LUT_LEG = run_experiment(
    mean_t_LEG,
    'relative',     # "relative" or "LS"
    err_regstatk10_large,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'LEG_err_regstatk10_large')

np.save(dir_path + "regstatk10_large_LUT_LEG_n_{n}.npy", regstatk10_large_LUT_LEG)
np.save(dir_path +"AMG_LUT" /f"regstatk10_large_AMG_LUT_LEG_n_{n}.npy", regstatk10_large_AMG_LUT_LEG)


regstatk10_large_LUT_PRO, regstatk10_large_AMG_LUT_PRO = run_experiment(
    mean_t_PRO,
    'relative',     # "relative" or "LS"
    err_regstatk10_large,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'PRO_err_regstatk10_large')

np.save(dir_path + "regstatk10_large_LUT_PRO_n_{n}.npy", regstatk10_large_LUT_PRO)
np.save(dir_path +"AMG_LUT" /f"regstatk10_large_AMG_LUT_PRO_n_{n}.npy", regstatk10_large_AMG_LUT_PRO)

err_regstatk10_largeRE = err_rel_dict(err_r=0.06, err_I=err_rel_I_Yp, err_QC0=0.2, err_T=0.01, err_else=0.1)

regstatk10_largeRE_LUT_LEG, regstatk10_largeRE_AMG_LUT_LEG = run_experiment(
    mean_t_LEG,
    'relative',     # "relative" or "LS"
    err_regstatk10_largeRE,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'LEG_err_regstatk10_largeRE')

np.save(dir_path + "regstatk10_largeRE_LUT_LEG_n_{n}.npy", regstatk10_largeRE_LUT_LEG)
np.save(dir_path +"AMG_LUT" /f"regstatk10_largeRE_AMG_LUT_LEG_n_{n}.npy", regstatk10_largeRE_AMG_LUT_LEG)


regstatk10_largeRE_LUT_PRO, regstatk10_largeRE_AMG_LUT_PRO = run_experiment(
    mean_t_PRO,
    'relative',     # "relative" or "LS"
    err_regstatk10_largeRE,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'PRO_err_regstatk10_largeRE')

np.save(dir_path + "regstatk10_largeRE_LUT_PRO_n_{n}.npy", regstatk10_largeRE_LUT_PRO)
np.save(dir_path +"AMG_LUT" /f"regstatk10_largeRE_AMG_LUT_PRO_n_{n}.npy", regstatk10_largeRE_AMG_LUT_PRO)

####### for all scenario using the std of SOC stock measurement as initial error (we previously took the relative uncertainty to keep same structure)
LEG_err_opti_mesRE = err_rel_dict(err_r=0.06, err_I=err_rel_I_DAM, err_QC0=err_rel_QC0_LEG, err_T=0.01, err_else=0.1)
PRO_err_opti_mesRE = err_rel_dict(err_r=0.06, err_I=err_rel_I_DAM, err_QC0=err_rel_QC0_PRO, err_T=0.01, err_else=0.1)

opti_mesRE_LUT_LEG, opti_mesRE_AMG_LUT_LEG = run_experiment(
    mean_t_LEG,
    'relative',     # "relative" or "LS"
    LEG_err_opti_mesRE,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'LEG_err_opti_mesRE')

np.save(dir_path + "opti_mesRE_LUT_LEG_n_{n}.npy", opti_mesRE_LUT_LEG)
np.save(dir_path +"AMG_LUT" /f"opti_mesRE_AMG_LUT_LEG_n_{n}.npy", opti_mesRE_AMG_LUT_LEG)


opti_mesRE_LUT_PRO, opti_mesRE_AMG_LUT_PRO = run_experiment(
    mean_t_PRO,
    'relative',     # "relative" or "LS"
    PRO_err_opti_mesRE,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'PRO_err_opti_mesRE')

np.save(dir_path + "opti_mesRE_LUT_PRO_n_{n}.npy", opti_mesRE_LUT_PRO)
np.save(dir_path +"AMG_LUT" /f"opti_mesRE_AMG_LUT_PRO_n_{n}.npy", opti_mesRE_AMG_LUT_PRO)

LEG_err_opti_mesdef = err_rel_dict(err_r=0.18, err_I=err_rel_I_DAM, err_QC0=err_rel_QC0_LEG, err_T=0.01, err_else=0.1)
PRO_err_opti_mesdef = err_rel_dict(err_r=0.18, err_I=err_rel_I_DAM, err_QC0=err_rel_QC0_PRO, err_T=0.01, err_else=0.1)



opti_mesdef_LUT_LEG, opti_mesdef_AMG_LUT_LEG = run_experiment(
    mean_t_LEG,
    'relative',     # "relative" or "LS"
    LEG_err_opti_mesdef,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'LEG_err_opti_mesdef')

np.save(dir_path + "opti_mesdef_LUT_LEG_n_{n}.npy", opti_mesdef_LUT_LEG)
np.save(dir_path +"AMG_LUT" /f"opti_mesdef_AMG_LUT_LEG_n_{n}.npy", opti_mesdef_AMG_LUT_LEG)


opti_mesdef_LUT_PRO, opti_mesdef_AMG_LUT_PRO = run_experiment(
    mean_t_PRO,
    'relative',     # "relative" or "LS"
    PRO_err_opti_mesdef,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'PRO_err_opti_mesdef')

np.save(dir_path + "opti_mesdef_LUT_PRO_n_{n}.npy", opti_mesdef_LUT_PRO)
np.save(dir_path +"AMG_LUT" /f"opti_mesdef_AMG_LUT_PRO_n_{n}.npy", opti_mesdef_AMG_LUT_PRO)


LEG_err_pessim_mesRE = err_rel_dict(err_r=0.06, err_I=err_rel_I_Yp, err_QC0=err_rel_QC0_LEG, err_T=0.01, err_else=0.1)
PRO_err_pessim_mesRE = err_rel_dict(err_r=0.06, err_I=err_rel_I_Yp, err_QC0=err_rel_QC0_PRO, err_T=0.01, err_else=0.1)

pessim_mesRE_LUT_LEG, pessim_mesRE_AMG_LUT_LEG = run_experiment(
    mean_t_LEG,
    'LS',     # "relative" or "LS"
    LEG_err_pessim_mesRE,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'LEG_err_pessim_mesRE')

np.save(dir_path + "pessim_mesRE_LUT_LEG_n_{n}.npy", pessim_mesRE_LUT_LEG)
np.save(dir_path +"AMG_LUT" /f"pessim_mesRE_AMG_LUT_LEG_n_{n}.npy", pessim_mesRE_AMG_LUT_LEG)


pessim_mesRE_LUT_PRO, pessim_mesRE_AMG_LUT_PRO = run_experiment(
    mean_t_PRO,
    'LS',     # "relative" or "LS"
    PRO_err_pessim_mesRE,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'PRO_err_pessim_mesRE')

np.save(dir_path + "pessim_mesRE_LUT_PRO_n_{n}.npy", pessim_mesRE_LUT_PRO)
np.save(dir_path +"AMG_LUT" /f"pessim_mesRE_AMG_LUT_PRO_n_{n}.npy", pessim_mesRE_AMG_LUT_PRO)


LEG_err_pessim_mesdef = err_rel_dict(err_r=0.18, err_I=err_rel_I_Yp, err_QC0=err_rel_QC0_LEG, err_T=0.01, err_else=0.1)
PRO_err_pessim_mesdef = err_rel_dict(err_r=0.18, err_I=err_rel_I_Yp, err_QC0=err_rel_QC0_PRO, err_T=0.01, err_else=0.1)

pessim_mesdef_LUT_LEG, pessim_mesdef_AMG_LUT_LEG = run_experiment(
    mean_t_LEG,
    'LS',     # "relative" or "LS"
    LEG_err_pessim_mesdef,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'LEG_err_pessim_mesdef')

np.save(dir_path + "pessim_mesdef_LUT_LEG_n_{n}.npy", pessim_mesdef_LUT_LEG)
np.save(dir_path +"AMG_LUT" /f"pessim_mesdef_AMG_LUT_LEG_n_{n}.npy", pessim_mesdef_AMG_LUT_LEG)


pessim_mesdef_LUT_PRO, pessim_mesdef_AMG_LUT_PRO = run_experiment(
    mean_t_PRO,
    'LS',     # "relative" or "LS"
    PRO_err_pessim_mesdef,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    'PRO_err_pessim_mesdef')

np.save(dir_path + "pessim_mesdef_LUT_PRO_n_{n}.npy", pessim_mesdef_LUT_PRO)
np.save(dir_path +"AMG_LUT" /f"pessim_mesdef_AMG_LUT_PRO_n_{n}.npy", pessim_mesdef_AMG_LUT_PRO)

#%% NUMERICAL SAMPLING WITH DIFFERENT SAMPLE SIZE n FOR SECTION 3.1. Comparison of the analytical based method with numerical sampling


def run_experiment(
    mean_t,
    error_k_model,     # "relative" or "LS"
    err_rel,    # dict
    sigma_LS,
    param,
    t,
    n,
    keys,
    dir_path_sigma_t,
    name_conf_err
    
    
):

    # 1. Build sigma_t
    if error_k_model == "relative":
        sigma_t = input_err_rel(mean_t, err_rel)
        
    elif error_k_model == "LS":
        sigma_t = input_err_LS(mean_t, sigma_LS, err_rel)
        
    else:
        raise ValueError("Unknown error model")

    # 2. Generate LUT
    LUT = generate_LUT(mean_t, sigma_t, keys, t, n)

    # 3. Run AMG
    AMG_LUT = AMG(LUT,t,n)

    return LUT, AMG_LUT

keys = ['T', 'H', 'Clay', 'CaCO3', 'pH', 'C/N', 'I', 'QC0', 'r']

### We create numerical sampling with different sampling size n
### For each sampling size n, we create five simulations s,
    ### to check the convergence of numerical sampling within a given sampling size
    
for n in [100, 1000, 10000, 100000]:
    for s in ['s1', 's2', 's3', 's4', 's5']:
        
    
        ### organised files for results with straw export
        dir_path = f"../outputs/MC_size_effect/{s}_{n}_LUT/"
        

        dir_path = Path(dir_path)
        
        (dir_path / "AMG_LUT").mkdir(parents=True, exist_ok=True)
        

            
        err_opti_local = err_rel_dict(err_r=0.06, err_I=err_rel_I_DAM, err_QC0=0.1, err_T=0.01, err_else=0.1)
        
        opti_local_LUT_LEG, opti_local_AMG_LUT_LEG = run_experiment(
            mean_t_LEG,
            'relative',     # "relative" or "LS"
            err_opti_local,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'LEG_err_opti_local')
        
        np.save(dir_path / f"opti_local_LUT_LEG_n_{n}.npy", opti_local_LUT_LEG)
        np.save(dir_path /"AMG_LUT" /f"opti_local_AMG_LUT_LEG_n_{n}.npy", opti_local_AMG_LUT_LEG)
        
        
        opti_local_LUT_PRO, opti_local_AMG_LUT_PRO = run_experiment(
            mean_t_PRO,
            'relative',     # "relative" or "LS"
            err_opti_local,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'PRO_err_opti_local')
        
        np.save(dir_path / f"opti_local_LUT_PRO_n_{n}.npy", opti_local_LUT_PRO)
        np.save(dir_path /"AMG_LUT" /f"opti_local_AMG_LUT_PRO_n_{n}.npy", opti_local_AMG_LUT_PRO)
        
        err_opti_localdef = err_rel_dict(err_r=0.18, err_I=err_rel_I_DAM, err_QC0=0.1, err_T=0.01, err_else=0.1)
        
        opti_localdef_LUT_LEG, opti_localdef_AMG_LUT_LEG = run_experiment(
            mean_t_LEG,
            'relative',     # "relative" or "LS"
            err_opti_localdef,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'LEG_err_opti_localdef')
        
        np.save(dir_path / f"opti_localdef_LUT_LEG_n_{n}.npy", opti_localdef_LUT_LEG)
        np.save(dir_path /"AMG_LUT" /f"opti_localdef_AMG_LUT_LEG_n_{n}.npy", opti_localdef_AMG_LUT_LEG)
        
        
        opti_localdef_LUT_PRO, opti_localdef_AMG_LUT_PRO = run_experiment(
            mean_t_PRO,
            'relative',     # "relative" or "LS"
            err_opti_localdef,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'PRO_err_opti_localdef')
        
        np.save(dir_path / f"opti_localdef_LUT_PRO_n_{n}.npy", opti_localdef_LUT_PRO)
        np.save(dir_path /"AMG_LUT" /f"opti_localdef_AMG_LUT_PRO_n_{n}.npy", opti_localdef_AMG_LUT_PRO)
        
        
        
        
        err_opti_large = err_rel_dict(err_r=0.18, err_I=err_rel_I_DAM, err_QC0=0.2, err_T=0.01, err_else=0.1)
        
        opti_large_LUT_LEG, opti_large_AMG_LUT_LEG = run_experiment(
            mean_t_LEG,
            'relative',     # "relative" or "LS"
            err_opti_large,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'LEG_err_opti_large')
        
        np.save(dir_path / f"opti_large_LUT_LEG_n_{n}.npy", opti_large_LUT_LEG)
        np.save(dir_path /"AMG_LUT" /f"opti_large_AMG_LUT_LEG_n_{n}.npy", opti_large_AMG_LUT_LEG)
        
        
        opti_large_LUT_PRO, opti_large_AMG_LUT_PRO = run_experiment(
            mean_t_PRO,
            'relative',     # "relative" or "LS"
            err_opti_large,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'PRO_err_opti_large')
        
        np.save(dir_path / f"opti_large_LUT_PRO_n_{n}.npy", opti_large_LUT_PRO)
        np.save(dir_path /"AMG_LUT" /f"opti_large_AMG_LUT_PRO_n_{n}.npy", opti_large_AMG_LUT_PRO)
        
        
        err_opti_largeRE = err_rel_dict(err_r=0.06, err_I=err_rel_I_DAM, err_QC0=0.2, err_T=0.01, err_else=0.1)
        
        opti_largeRE_LUT_LEG, opti_largeRE_AMG_LUT_LEG = run_experiment(
            mean_t_LEG,
            'relative',     # "relative" or "LS"
            err_opti_largeRE,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'LEG_err_opti_largeRE')
        
        np.save(dir_path / f"opti_largeRE_LUT_LEG_n_{n}.npy", opti_largeRE_LUT_LEG)
        np.save(dir_path /"AMG_LUT" /f"opti_largeRE_AMG_LUT_LEG_n_{n}.npy", opti_largeRE_AMG_LUT_LEG)
        
        
        opti_largeRE_LUT_PRO, opti_largeRE_AMG_LUT_PRO = run_experiment(
            mean_t_PRO,
            'relative',     # "relative" or "LS"
            err_opti_largeRE,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'PRO_err_opti_largeRE')
        
        np.save(dir_path / f"opti_largeRE_LUT_PRO_n_{n}.npy", opti_largeRE_LUT_PRO)
        np.save(dir_path /"AMG_LUT" /f"opti_largeRE_AMG_LUT_PRO_n_{n}.npy", opti_largeRE_AMG_LUT_PRO)
        
        
        ## as we will use LS error, we don't care about the value of err_else et err_T
        
        err_pessim_large = err_rel_dict(err_r=0.18, err_I=err_rel_I_Yp, err_QC0=0.2, err_T=0.01, err_else=0.1)
        
        pessim_large_LUT_LEG, pessim_large_AMG_LUT_LEG = run_experiment(
            mean_t_LEG,
            'LS',     # "relative" or "LS"
            err_pessim_large,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'LEG_err_pessim_large')
        
        np.save(dir_path / f"pessim_large_LUT_LEG_n_{n}.npy", pessim_large_LUT_LEG)
        np.save(dir_path /"AMG_LUT" /f"pessim_large_AMG_LUT_LEG_n_{n}.npy", pessim_large_AMG_LUT_LEG)
        
        
        pessim_large_LUT_PRO, pessim_large_AMG_LUT_PRO = run_experiment(
            mean_t_PRO,
            'LS',     # "relative" or "LS"
            err_pessim_large,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'PRO_err_pessim_large')
        
        np.save(dir_path / f"pessim_large_LUT_PRO_n_{n}.npy", pessim_large_LUT_PRO)
        np.save(dir_path /"AMG_LUT" /f"pessim_large_AMG_LUT_PRO_n_{n}.npy", pessim_large_AMG_LUT_PRO)
        
        
        ## as we will use LS error, we don't care about the value of err_else et err_T
        
        err_pessim_largeRE = err_rel_dict(err_r=0.06, err_I=err_rel_I_Yp, err_QC0=0.2, err_T=0.01, err_else=0.1)
        
        pessim_largeRE_LUT_LEG, pessim_largeRE_AMG_LUT_LEG = run_experiment(
            mean_t_LEG,
            'LS',     # "relative" or "LS"
            err_pessim_largeRE,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'LEG_err_pessim_largeRE')
        
        np.save(dir_path / f"pessim_largeRE_LUT_LEG_n_{n}.npy", pessim_largeRE_LUT_LEG)
        np.save(dir_path /"AMG_LUT" /f"pessim_largeRE_AMG_LUT_LEG_n_{n}.npy", pessim_largeRE_AMG_LUT_LEG)
        
        
        pessim_largeRE_LUT_PRO, pessim_largeRE_AMG_LUT_PRO = run_experiment(
            mean_t_PRO,
            'LS',     # "relative" or "LS"
            err_pessim_largeRE,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'PRO_err_pessim_largeRE')
        
        np.save(dir_path / f"pessim_largeRE_LUT_PRO_n_{n}.npy", pessim_largeRE_LUT_PRO)
        np.save(dir_path /"AMG_LUT" /f"pessim_largeRE_AMG_LUT_PRO_n_{n}.npy", pessim_largeRE_AMG_LUT_PRO)
        
        
        err_pessim_local = err_rel_dict(err_r=0.06, err_I=err_rel_I_Yp, err_QC0=0.1, err_T=0.01, err_else=0.1)
        
        pessim_local_LUT_LEG, pessim_local_AMG_LUT_LEG = run_experiment(
            mean_t_LEG,
            'LS',     # "relative" or "LS"
            err_pessim_local,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'LEG_err_pessim_local')
        
        np.save(dir_path / f"pessim_local_LUT_LEG_n_{n}.npy", pessim_local_LUT_LEG)
        np.save(dir_path /"AMG_LUT" /f"pessim_local_AMG_LUT_LEG_n_{n}.npy", pessim_local_AMG_LUT_LEG)
        
        
        pessim_local_LUT_PRO, pessim_local_AMG_LUT_PRO = run_experiment(
            mean_t_PRO,
            'LS',     # "relative" or "LS"
            err_pessim_local,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'PRO_err_pessim_local')
        
        np.save(dir_path / f"pessim_local_LUT_PRO_n_{n}.npy", pessim_local_LUT_PRO)
        np.save(dir_path /"AMG_LUT" /f"pessim_local_AMG_LUT_PRO_n_{n}.npy", pessim_local_AMG_LUT_PRO)
        
        
        err_pessim_localdef = err_rel_dict(err_r=0.18, err_I=err_rel_I_Yp, err_QC0=0.1, err_T=0.01, err_else=0.1)
        
        pessim_localdef_LUT_LEG, pessim_localdef_AMG_LUT_LEG = run_experiment(
            mean_t_LEG,
            'LS',     # "relative" or "LS"
            err_pessim_localdef,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'LEG_err_pessim_localdef')
        
        np.save(dir_path / f"pessim_localdef_LUT_LEG_n_{n}.npy", pessim_localdef_LUT_LEG)
        np.save(dir_path /"AMG_LUT" /f"pessim_localdef_AMG_LUT_LEG_n_{n}.npy", pessim_localdef_AMG_LUT_LEG)
        
        
        pessim_localdef_LUT_PRO, pessim_localdef_AMG_LUT_PRO = run_experiment(
            mean_t_PRO,
            'LS',     # "relative" or "LS"
            err_pessim_localdef,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'PRO_err_pessim_localdef')
        
        np.save(dir_path / f"pessim_localdef_LUT_PRO_n_{n}.npy", pessim_localdef_LUT_PRO)
        np.save(dir_path /"AMG_LUT" /f"pessim_localdef_AMG_LUT_PRO_n_{n}.npy", pessim_localdef_AMG_LUT_PRO)
        

        
        
        err_ACEOkLS_localdef = err_rel_dict(err_r=0.18, err_I=err_rel_I_DAM, err_QC0=0.1, err_T=0.01, err_else=0.1)
        
        ACEOkLS_localdef_LUT_LEG, ACEOkLS_localdef_AMG_LUT_LEG = run_experiment(
            mean_t_LEG,
            'LS',     # "relative" or "LS"
            err_ACEOkLS_localdef,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'LEG_err_ACEOkLS_localdef')
        
        np.save(dir_path / f"ACEOkLS_localdef_LUT_LEG_n_{n}.npy", ACEOkLS_localdef_LUT_LEG)
        np.save(dir_path /"AMG_LUT" /f"ACEOkLS_localdef_AMG_LUT_LEG_n_{n}.npy", ACEOkLS_localdef_AMG_LUT_LEG)
        
        
        ACEOkLS_localdef_LUT_PRO, ACEOkLS_localdef_AMG_LUT_PRO = run_experiment(
            mean_t_PRO,
            'LS',     # "relative" or "LS"
            err_ACEOkLS_localdef,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'PRO_err_ACEOkLS_localdef')
        
        np.save(dir_path / f"ACEOkLS_localdef_LUT_PRO_n_{n}.npy", ACEOkLS_localdef_LUT_PRO)
        np.save(dir_path /"AMG_LUT" /f"ACEOkLS_localdef_AMG_LUT_PRO_n_{n}.npy", ACEOkLS_localdef_AMG_LUT_PRO)
        
        
        err_ACEOkLS_local = err_rel_dict(err_r=0.06, err_I=err_rel_I_DAM, err_QC0=0.1, err_T=0.01, err_else=0.1)
        
        ACEOkLS_local_LUT_LEG, ACEOkLS_local_AMG_LUT_LEG = run_experiment(
            mean_t_LEG,
            'LS',     # "relative" or "LS"
            err_ACEOkLS_local,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'LEG_err_ACEOkLS_local')
        
        np.save(dir_path / f"ACEOkLS_local_LUT_LEG_n_{n}.npy", ACEOkLS_local_LUT_LEG)
        np.save(dir_path /"AMG_LUT" /f"ACEOkLS_local_AMG_LUT_LEG_n_{n}.npy", ACEOkLS_local_AMG_LUT_LEG)
        
        
        ACEOkLS_local_LUT_PRO, ACEOkLS_local_AMG_LUT_PRO = run_experiment(
            mean_t_PRO,
            'LS',     # "relative" or "LS"
            err_ACEOkLS_local,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'PRO_err_ACEOkLS_local')
        
        np.save(dir_path / f"ACEOkLS_local_LUT_PRO_n_{n}.npy", ACEOkLS_local_LUT_PRO)
        np.save(dir_path /"AMG_LUT" /f"ACEOkLS_local_AMG_LUT_PRO_n_{n}.npy", ACEOkLS_local_AMG_LUT_PRO)
        
        
        err_ACEOkLS_large = err_rel_dict(err_r=0.18, err_I=err_rel_I_DAM, err_QC0=0.2, err_T=0.01, err_else=0.1)
        
        ACEOkLS_large_LUT_LEG, ACEOkLS_large_AMG_LUT_LEG = run_experiment(
            mean_t_LEG,
            'LS',     # "relative" or "LS"
            err_ACEOkLS_large,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'LEG_err_ACEOkLS_large')
        
        np.save(dir_path / f"ACEOkLS_large_LUT_LEG_n_{n}.npy", ACEOkLS_large_LUT_LEG)
        np.save(dir_path /"AMG_LUT" /f"ACEOkLS_large_AMG_LUT_LEG_n_{n}.npy", ACEOkLS_large_AMG_LUT_LEG)
        
        
        ACEOkLS_large_LUT_PRO, ACEOkLS_large_AMG_LUT_PRO = run_experiment(
            mean_t_PRO,
            'LS',     # "relative" or "LS"
            err_ACEOkLS_large,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'PRO_err_ACEOkLS_large')
        
        np.save(dir_path / f"ACEOkLS_large_LUT_PRO_n_{n}.npy", ACEOkLS_large_LUT_PRO)
        np.save(dir_path /"AMG_LUT" /f"ACEOkLS_large_AMG_LUT_PRO_n_{n}.npy", ACEOkLS_large_AMG_LUT_PRO)
        
        
        err_ACEOkLS_largeRE = err_rel_dict(err_r=0.06, err_I=err_rel_I_DAM, err_QC0=0.2, err_T=0.01, err_else=0.1)
        
        ACEOkLS_largeRE_LUT_LEG, ACEOkLS_largeRE_AMG_LUT_LEG = run_experiment(
            mean_t_LEG,
            'LS',     # "relative" or "LS"
            err_ACEOkLS_largeRE,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'LEG_err_ACEOkLS_largeRE')
        
        np.save(dir_path / f"ACEOkLS_largeRE_LUT_LEG_n_{n}.npy", ACEOkLS_largeRE_LUT_LEG)
        np.save(dir_path /"AMG_LUT" /f"ACEOkLS_largeRE_AMG_LUT_LEG_n_{n}.npy", ACEOkLS_largeRE_AMG_LUT_LEG)
        
        
        ACEOkLS_largeRE_LUT_PRO, ACEOkLS_largeRE_AMG_LUT_PRO = run_experiment(
            mean_t_PRO,
            'LS',     # "relative" or "LS"
            err_ACEOkLS_largeRE,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'PRO_err_ACEOkLS_largeRE')
        
        np.save(dir_path / f"ACEOkLS_largeRE_LUT_PRO_n_{n}.npy", ACEOkLS_largeRE_LUT_PRO)
        np.save(dir_path /"AMG_LUT" /f"ACEOkLS_largeRE_AMG_LUT_PRO_n_{n}.npy", ACEOkLS_largeRE_AMG_LUT_PRO)
        
        
        
        err_regstatk10_localdef = err_rel_dict(err_r=0.18, err_I=err_rel_I_Yp, err_QC0=0.1, err_T=0.01, err_else=0.1)
        
        regstatk10_localdef_LUT_LEG, regstatk10_localdef_AMG_LUT_LEG = run_experiment(
            mean_t_LEG,
            'relative',     # "relative" or "LS"
            err_regstatk10_localdef,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'LEG_err_regstatk10_localdef')
        
        np.save(dir_path / f"regstatk10_localdef_LUT_LEG_n_{n}.npy", regstatk10_localdef_LUT_LEG)
        np.save(dir_path /"AMG_LUT" /f"regstatk10_localdef_AMG_LUT_LEG_n_{n}.npy", regstatk10_localdef_AMG_LUT_LEG)
        
        
        regstatk10_localdef_LUT_PRO, regstatk10_localdef_AMG_LUT_PRO = run_experiment(
            mean_t_PRO,
            'relative',     # "relative" or "LS"
            err_regstatk10_localdef,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'PRO_err_regstatk10_localdef')
        
        np.save(dir_path / f"regstatk10_localdef_LUT_PRO_n_{n}.npy", regstatk10_localdef_LUT_PRO)
        np.save(dir_path /"AMG_LUT" /f"regstatk10_localdef_AMG_LUT_PRO_n_{n}.npy", regstatk10_localdef_AMG_LUT_PRO)
        
        
        err_regstatk10_local = err_rel_dict(err_r=0.06, err_I=err_rel_I_Yp, err_QC0=0.1, err_T=0.01, err_else=0.1)
        
        regstatk10_local_LUT_LEG, regstatk10_local_AMG_LUT_LEG = run_experiment(
            mean_t_LEG,
            'relative',     # "relative" or "LS"
            err_regstatk10_local,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'LEG_err_regstatk10_local')
        
        np.save(dir_path / f"regstatk10_local_LUT_LEG_n_{n}.npy", regstatk10_local_LUT_LEG)
        np.save(dir_path /"AMG_LUT" /f"regstatk10_local_AMG_LUT_LEG_n_{n}.npy", regstatk10_local_AMG_LUT_LEG)
        
        
        regstatk10_local_LUT_PRO, regstatk10_local_AMG_LUT_PRO = run_experiment(
            mean_t_PRO,
            'relative',     # "relative" or "LS"
            err_regstatk10_local,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'PRO_err_regstatk10_local')
        
        np.save(dir_path / f"regstatk10_local_LUT_PRO_n_{n}.npy", regstatk10_local_LUT_PRO)
        np.save(dir_path /"AMG_LUT" /f"regstatk10_local_AMG_LUT_PRO_n_{n}.npy", regstatk10_local_AMG_LUT_PRO)
        
        
        err_regstatk10_large = err_rel_dict(err_r=0.18, err_I=err_rel_I_Yp, err_QC0=0.2, err_T=0.01, err_else=0.1)
        
        regstatk10_large_LUT_LEG, regstatk10_large_AMG_LUT_LEG = run_experiment(
            mean_t_LEG,
            'relative',     # "relative" or "LS"
            err_regstatk10_large,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'LEG_err_regstatk10_large')
        
        np.save(dir_path / f"regstatk10_large_LUT_LEG_n_{n}.npy", regstatk10_large_LUT_LEG)
        np.save(dir_path /"AMG_LUT" /f"regstatk10_large_AMG_LUT_LEG_n_{n}.npy", regstatk10_large_AMG_LUT_LEG)
        
        
        regstatk10_large_LUT_PRO, regstatk10_large_AMG_LUT_PRO = run_experiment(
            mean_t_PRO,
            'relative',     # "relative" or "LS"
            err_regstatk10_large,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'PRO_err_regstatk10_large')
        
        np.save(dir_path / f"regstatk10_large_LUT_PRO_n_{n}.npy", regstatk10_large_LUT_PRO)
        np.save(dir_path /"AMG_LUT" /f"regstatk10_large_AMG_LUT_PRO_n_{n}.npy", regstatk10_large_AMG_LUT_PRO)
        
        
        err_regstatk10_largeRE = err_rel_dict(err_r=0.06, err_I=err_rel_I_Yp, err_QC0=0.2, err_T=0.01, err_else=0.1)
        
        regstatk10_largeRE_LUT_LEG, regstatk10_largeRE_AMG_LUT_LEG = run_experiment(
            mean_t_LEG,
            'relative',     # "relative" or "LS"
            err_regstatk10_largeRE,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'LEG_err_regstatk10_largeRE')
        
        np.save(dir_path / f"regstatk10_largeRE_LUT_LEG_n_{n}.npy", regstatk10_largeRE_LUT_LEG)
        np.save(dir_path /"AMG_LUT" /f"regstatk10_largeRE_AMG_LUT_LEG_n_{n}.npy", regstatk10_largeRE_AMG_LUT_LEG)
        
        
        regstatk10_largeRE_LUT_PRO, regstatk10_largeRE_AMG_LUT_PRO = run_experiment(
            mean_t_PRO,
            'relative',     # "relative" or "LS"
            err_regstatk10_largeRE,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'PRO_err_regstatk10_largeRE')
        
        np.save(dir_path / f"regstatk10_largeRE_LUT_PRO_n_{n}.npy", regstatk10_largeRE_LUT_PRO)
        np.save(dir_path /"AMG_LUT" /f"regstatk10_largeRE_AMG_LUT_PRO_n_{n}.npy", regstatk10_largeRE_AMG_LUT_PRO)
        
        ####### for all scenario using the std of SOC stock measurement as initial error (we previously took the relative uncertainty to keep same structure)
        LEG_err_opti_mesRE = err_rel_dict(err_r=0.06, err_I=err_rel_I_DAM, err_QC0=err_rel_QC0_LEG, err_T=0.01, err_else=0.1)
        PRO_err_opti_mesRE = err_rel_dict(err_r=0.06, err_I=err_rel_I_DAM, err_QC0=err_rel_QC0_PRO, err_T=0.01, err_else=0.1)
        
        opti_mesRE_LUT_LEG, opti_mesRE_AMG_LUT_LEG = run_experiment(
            mean_t_LEG,
            'relative',     # "relative" or "LS"
            LEG_err_opti_mesRE,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'LEG_err_opti_mesRE')
        
        np.save(dir_path / f"opti_mesRE_LUT_LEG_n_{n}.npy", opti_mesRE_LUT_LEG)
        np.save(dir_path /"AMG_LUT" /f"opti_mesRE_AMG_LUT_LEG_n_{n}.npy", opti_mesRE_AMG_LUT_LEG)
        
        
        opti_mesRE_LUT_PRO, opti_mesRE_AMG_LUT_PRO = run_experiment(
            mean_t_PRO,
            'relative',     # "relative" or "LS"
            PRO_err_opti_mesRE,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'PRO_err_opti_mesRE')
        
        np.save(dir_path / f"opti_mesRE_LUT_PRO_n_{n}.npy", opti_mesRE_LUT_PRO)
        np.save(dir_path /"AMG_LUT" /f"opti_mesRE_AMG_LUT_PRO_n_{n}.npy", opti_mesRE_AMG_LUT_PRO)
        
        LEG_err_opti_mesdef = err_rel_dict(err_r=0.18, err_I=err_rel_I_DAM, err_QC0=err_rel_QC0_LEG, err_T=0.01, err_else=0.1)
        PRO_err_opti_mesdef = err_rel_dict(err_r=0.18, err_I=err_rel_I_DAM, err_QC0=err_rel_QC0_PRO, err_T=0.01, err_else=0.1)
        
        
        
        opti_mesdef_LUT_LEG, opti_mesdef_AMG_LUT_LEG = run_experiment(
            mean_t_LEG,
            'relative',     # "relative" or "LS"
            LEG_err_opti_mesdef,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'LEG_err_opti_mesdef')
        
        np.save(dir_path / f"opti_mesdef_LUT_LEG_n_{n}.npy", opti_mesdef_LUT_LEG)
        np.save(dir_path /"AMG_LUT" /f"opti_mesdef_AMG_LUT_LEG_n_{n}.npy", opti_mesdef_AMG_LUT_LEG)
        
        
        opti_mesdef_LUT_PRO, opti_mesdef_AMG_LUT_PRO = run_experiment(
            mean_t_PRO,
            'relative',     # "relative" or "LS"
            PRO_err_opti_mesdef,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'PRO_err_opti_mesdef')
        
        np.save(dir_path / f"opti_mesdef_LUT_PRO_n_{n}.npy", opti_mesdef_LUT_PRO)
        np.save(dir_path /"AMG_LUT" /f"opti_mesdef_AMG_LUT_PRO_n_{n}.npy", opti_mesdef_AMG_LUT_PRO)
        
        
        
        
        
        LEG_err_pessim_mesRE = err_rel_dict(err_r=0.06, err_I=err_rel_I_Yp, err_QC0=err_rel_QC0_LEG, err_T=0.01, err_else=0.1)
        PRO_err_pessim_mesRE = err_rel_dict(err_r=0.06, err_I=err_rel_I_Yp, err_QC0=err_rel_QC0_PRO, err_T=0.01, err_else=0.1)
        
        pessim_mesRE_LUT_LEG, pessim_mesRE_AMG_LUT_LEG = run_experiment(
            mean_t_LEG,
            'LS',     # "relative" or "LS"
            LEG_err_pessim_mesRE,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'LEG_err_pessim_mesRE')
        
        np.save(dir_path / f"pessim_mesRE_LUT_LEG_n_{n}.npy", pessim_mesRE_LUT_LEG)
        np.save(dir_path /"AMG_LUT" /f"pessim_mesRE_AMG_LUT_LEG_n_{n}.npy", pessim_mesRE_AMG_LUT_LEG)
        
        
        pessim_mesRE_LUT_PRO, pessim_mesRE_AMG_LUT_PRO = run_experiment(
            mean_t_PRO,
            'LS',     # "relative" or "LS"
            PRO_err_pessim_mesRE,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'PRO_err_pessim_mesRE')
        
        np.save(dir_path / f"pessim_mesRE_LUT_PRO_n_{n}.npy", pessim_mesRE_LUT_PRO)
        np.save(dir_path /"AMG_LUT" /f"pessim_mesRE_AMG_LUT_PRO_n_{n}.npy", pessim_mesRE_AMG_LUT_PRO)
        
        
        LEG_err_pessim_mesdef = err_rel_dict(err_r=0.18, err_I=err_rel_I_Yp, err_QC0=err_rel_QC0_LEG, err_T=0.01, err_else=0.1)
        PRO_err_pessim_mesdef = err_rel_dict(err_r=0.18, err_I=err_rel_I_Yp, err_QC0=err_rel_QC0_PRO, err_T=0.01, err_else=0.1)
        
        pessim_mesdef_LUT_LEG, pessim_mesdef_AMG_LUT_LEG = run_experiment(
            mean_t_LEG,
            'LS',     # "relative" or "LS"
            LEG_err_pessim_mesdef,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'LEG_err_pessim_mesdef')
        
        np.save(dir_path / f"pessim_mesdef_LUT_LEG_n_{n}.npy", pessim_mesdef_LUT_LEG)
        np.save(dir_path /"AMG_LUT" /f"pessim_mesdef_AMG_LUT_LEG_n_{n}.npy", pessim_mesdef_AMG_LUT_LEG)
        
        
        pessim_mesdef_LUT_PRO, pessim_mesdef_AMG_LUT_PRO = run_experiment(
            mean_t_PRO,
            'LS',     # "relative" or "LS"
            PRO_err_pessim_mesdef,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'PRO_err_pessim_mesdef')
        
        np.save(dir_path / f"pessim_mesdef_LUT_PRO_n_{n}.npy", pessim_mesdef_LUT_PRO)
        np.save(dir_path /"AMG_LUT" /f"pessim_mesdef_AMG_LUT_PRO_n_{n}.npy", pessim_mesdef_AMG_LUT_PRO)



        LEG_err_ACEOkLS_mesdef = err_rel_dict(err_r=0.18, err_I=err_rel_I_DAM, err_QC0=err_rel_QC0_LEG, err_T=0.01, err_else=0.1)
        PRO_err_ACEOkLS_mesdef = err_rel_dict(err_r=0.18, err_I=err_rel_I_DAM, err_QC0=err_rel_QC0_PRO, err_T=0.01, err_else=0.1)
        
        ACEOkLS_mesdef_LUT_LEG, ACEOkLS_mesdef_AMG_LUT_LEG = run_experiment(
            mean_t_LEG,
            'LS',     # "relative" or "LS"
            LEG_err_ACEOkLS_mesdef,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'LEG_err_ACEOkLS_mesdef')
        
        np.save(dir_path / f"ACEOkLS_mesdef_LUT_LEG_n_{n}.npy", ACEOkLS_mesdef_LUT_LEG)
        np.save(dir_path /"AMG_LUT" /f"ACEOkLS_mesdef_AMG_LUT_LEG_n_{n}.npy", ACEOkLS_mesdef_AMG_LUT_LEG)
        
        
        ACEOkLS_mesdef_LUT_PRO, ACEOkLS_mesdef_AMG_LUT_PRO = run_experiment(
            mean_t_PRO,
            'LS',     # "relative" or "LS"
            PRO_err_ACEOkLS_mesdef,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'PRO_err_ACEOkLS_mesdef')
        
        np.save(dir_path / f"ACEOkLS_mesdef_LUT_PRO_n_{n}.npy", ACEOkLS_mesdef_LUT_PRO)
        np.save(dir_path /"AMG_LUT" /f"ACEOkLS_mesdef_AMG_LUT_PRO_n_{n}.npy", ACEOkLS_mesdef_AMG_LUT_PRO)
        
                
        
        LEG_err_ACEOkLS_mesRE = err_rel_dict(err_r=0.06, err_I=err_rel_I_DAM, err_QC0=err_rel_QC0_LEG, err_T=0.01, err_else=0.1)
        PRO_err_ACEOkLS_mesRE = err_rel_dict(err_r=0.06, err_I=err_rel_I_DAM, err_QC0=err_rel_QC0_PRO, err_T=0.01, err_else=0.1)
        
        ACEOkLS_mesRE_LUT_LEG, ACEOkLS_mesRE_AMG_LUT_LEG = run_experiment(
            mean_t_LEG,
            'LS',     # "relative" or "LS"
            LEG_err_ACEOkLS_mesRE,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'LEG_err_ACEOkLS_mesRE')
        
        np.save(dir_path / f"ACEOkLS_mesRE_LUT_LEG_n_{n}.npy", ACEOkLS_mesRE_LUT_LEG)
        np.save(dir_path /"AMG_LUT" /f"ACEOkLS_mesRE_AMG_LUT_LEG_n_{n}.npy", ACEOkLS_mesRE_AMG_LUT_LEG)
        
        
        ACEOkLS_mesRE_LUT_PRO, ACEOkLS_mesRE_AMG_LUT_PRO = run_experiment(
            mean_t_PRO,
            'LS',     # "relative" or "LS"
            PRO_err_ACEOkLS_mesRE,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'PRO_err_ACEOkLS_mesRE')
        
        np.save(dir_path / f"ACEOkLS_mesRE_LUT_PRO_n_{n}.npy", ACEOkLS_mesRE_LUT_PRO)
        np.save(dir_path /"AMG_LUT" /f"ACEOkLS_mesRE_AMG_LUT_PRO_n_{n}.npy", ACEOkLS_mesRE_AMG_LUT_PRO)
        
        
        
        LEG_err_regstatk10_mesdef = err_rel_dict(err_r=0.18, err_I=err_rel_I_Yp, err_QC0=err_rel_QC0_LEG, err_T=0.01, err_else=0.1)
        PRO_err_regstatk10_mesdef = err_rel_dict(err_r=0.18, err_I=err_rel_I_Yp, err_QC0=err_rel_QC0_PRO, err_T=0.01, err_else=0.1)
        
        regstatk10_mesdef_LUT_LEG, regstatk10_mesdef_AMG_LUT_LEG = run_experiment(
            mean_t_LEG,
            'relative',     # "relative" or "LS"
            LEG_err_regstatk10_mesdef,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'LEG_err_regstatk10_mesdef')
        
        np.save(dir_path / f"regstatk10_mesdef_LUT_LEG_n_{n}.npy", regstatk10_mesdef_LUT_LEG)
        np.save(dir_path /"AMG_LUT" /f"regstatk10_mesdef_AMG_LUT_LEG_n_{n}.npy", regstatk10_mesdef_AMG_LUT_LEG)
        
        
        regstatk10_mesdef_LUT_PRO, regstatk10_mesdef_AMG_LUT_PRO = run_experiment(
            mean_t_PRO,
            'relative',     # "relative" or "LS"
            PRO_err_regstatk10_mesdef,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'PRO_err_regstatk10_mesdef')
        
        np.save(dir_path / f"regstatk10_mesdef_LUT_PRO_n_{n}.npy", regstatk10_mesdef_LUT_PRO)
        np.save(dir_path /"AMG_LUT" /f"regstatk10_mesdef_AMG_LUT_PRO_n_{n}.npy", regstatk10_mesdef_AMG_LUT_PRO)
        
                
        
        LEG_err_regstatk10_mesRE = err_rel_dict(err_r=0.06, err_I=err_rel_I_Yp, err_QC0=err_rel_QC0_LEG, err_T=0.01, err_else=0.1)
        PRO_err_regstatk10_mesRE = err_rel_dict(err_r=0.06, err_I=err_rel_I_Yp, err_QC0=err_rel_QC0_PRO, err_T=0.01, err_else=0.1)
        
        regstatk10_mesRE_LUT_LEG, regstatk10_mesRE_AMG_LUT_LEG = run_experiment(
            mean_t_LEG,
            'relative',     # "relative" or "LS"
            LEG_err_regstatk10_mesRE,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'LEG_err_regstatk10_mesRE')
        
        np.save(dir_path / f"regstatk10_mesRE_LUT_LEG_n_{n}.npy", regstatk10_mesRE_LUT_LEG)
        np.save(dir_path /"AMG_LUT" /f"regstatk10_mesRE_AMG_LUT_LEG_n_{n}.npy", regstatk10_mesRE_AMG_LUT_LEG)
        
        
        regstatk10_mesRE_LUT_PRO, regstatk10_mesRE_AMG_LUT_PRO = run_experiment(
            mean_t_PRO,
            'relative',     # "relative" or "LS"
            PRO_err_regstatk10_mesRE,    # dict
            sigma_LS,
            param,
            t,
            n,
            keys,
            dir_path_sigma_t,
            'PRO_err_regstatk10_mesRE')
        
        np.save(dir_path / f"regstatk10_mesRE_LUT_PRO_n_{n}.npy", regstatk10_mesRE_LUT_PRO)
        np.save(dir_path /"AMG_LUT" /f"regstatk10_mesRE_AMG_LUT_PRO_n_{n}.npy", regstatk10_mesRE_AMG_LUT_PRO)
