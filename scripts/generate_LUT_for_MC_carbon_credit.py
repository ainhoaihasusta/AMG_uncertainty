#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 19:52:24 2026

@author: lhasustaa
"""



import pandas as pd
import numpy as nps
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

# Create new management practice scenario, we take the exemple of a cover crops
# so we generate a new I_CF(t) representing the addition of carbon inputs from cover crops
# According to Wijmer et al. (2025) : the amount of carbon humified added to the soil by cover crops = 0.4 to 1.3 tC/ha

### Run once and save it to have the same sampling if need to update plot
#I_from_CC = np.random.uniform(0.4,1.3,t)

# sampling used in the paper
I_from_CC = np.load('../inputs/I_from_CC_sampling.npy')

# We assume in the carbon farming scenario (CF), that if the main crop is maize, there is no cover crops
# in rotation_filter, False corresponds to a maize year, and True to a non-maize year
rotation_filter =np.array([ True, False,  True, False,  True, False,  True, False,  True,
        True, False,  True, False,  True, False,  True, False,  True,
        True,  True, False,  True,  True,  True])

mean_t_LEG['I_CF'] =   mean_t_LEG['I'] + I_from_CC *rotation_filter# represent carbon inputs from Carbon farming scenario
mean_t_PRO['I_CF'] =   mean_t_PRO['I'] + I_from_CC *rotation_filter# represent carbon inputs from Carbon farming scenario


# To faciliate the calculation, we call DI(t) = I_CF(t) - I_B(t)
# mu_DI = mu_I_CF - mu_I_B
# as I_CF(t) and I_B(t) are independant, we have :
# sigma_DI² = sigma_I_CF² + sigma_I_B²
# we detailed everything but actually it's simpler : DI = I_from_CC*rotation_filter as we keep the same C_inp from the main crop in both scenario
mean_t_LEG['DI'] = mean_t_LEG['I_CF'] - mean_t_LEG['I']
mean_t_PRO['DI'] = mean_t_PRO['I_CF'] - mean_t_PRO['I']

# np.save('../inputs/mean_t_LEG.npy',mean_t_LEG)
# np.save('..inputs/mean_t_PRO.npy',mean_t_PRO)

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

#%% Definition of error values

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


##3 Error on initial stable/total SOC ratio r (absolute error, not relative)

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
"""

The name of the function corresponds to the different error scenarios used for the inputs variables, following this convention 'input_err_term1_term2' : 'term1' corresponds to the error scenario to compute carbon inputs uncertainty, and term2 for mineralisation rate uncertainty

- ACEO : when the scenario concerning carbon inputs uncertainty is Field-scale scenario ('ACEO' for using AgriCarbon-EO error),
- regstat : when the scenario concerning carbon inputs uncertainty is Large-scale scenario ('regstat' for using regional statistics error)
- kLS : when the error scenario for computing the mineralisation rate uncertainty is Large-scale scenario (kLS for Large-Scale error).
- k10 : when the error scenario for computing the mineralisation rate uncertainty is Field-scale scenario (k10 as in the field-scale scenario we assume a 10% relative error).
"""
    
def input_err_I_ACEO_k10(mean_t, err_rel=0.1, err_rel_I=err_rel_I_DAM):
    sigma_t = {}
    
    keys = [ 'T', 'H', 'Clay', 'pH', 'C/N']
    for k in keys : 
        sigma_t[k] = abs(err_rel*mean_t[k])
    sigma_t['CaCO3'] =  abs(np.random.normal(0,2.5,t)) #in Qualiagro, CaCO3=0, so we create a small random error
    sigma_t['I'] = abs(err_rel_I*mean_t['I'])
    sigma_t['I_CF'] =  err_rel_I* mean_t['I_CF'] # represent carbon inputs from Carbon farming scenario    
    sigma_t['DI'] = np.sqrt(sigma_t['I_CF']**2 + sigma_t['I']**2)
    return sigma_t


def input_err_I_ACEO_kLS(mean_t, sigma_LS, err_rel_I=err_rel_I_DAM): #LS for large scale ERA5 and LUCAS soil maps
    sigma_t = {}
    keys = [ 'T', 'H', 'Clay', 'CaCO3', 'pH', 'C/N']
    for k in keys :
        sigma_t[k]=sigma_LS[k]*np.ones(t) 
    sigma_t['I'] = abs(err_rel_I*mean_t['I'])
    sigma_t['I_CF'] =  err_rel_I* mean_t['I_CF'] # represent carbon inputs from Carbon farming scenario    
    sigma_t['DI'] = np.sqrt(sigma_t['I_CF']**2 + sigma_t['I']**2)

    return sigma_t

def input_err_I_regstat_kLS(mean_t, sigma_LS, err_rel_I=err_rel_I_Yp): #LS for large scale ERA5 and LUCAS soil maps
    sigma_t = {}
    keys = [ 'T', 'H', 'Clay', 'CaCO3', 'pH', 'C/N']
    for k in keys :
        sigma_t[k]=sigma_LS[k]*np.ones(t) 
    sigma_t['I'] = abs(err_rel_I*mean_t['I'])
    sigma_t['I_CF'] =  err_rel_I* mean_t['I_CF'] # represent carbon inputs from Carbon farming scenario    
    sigma_t['DI'] = np.sqrt(sigma_t['I_CF']**2 + sigma_t['I']**2)

    return sigma_t

def input_err_I_regstat_k10(mean_t, err_rel=0.1, err_rel_I=err_rel_I_Yp): #LS for large scale ERA5 and LUCAS soil maps
    sigma_t = {}
    keys = [ 'T', 'H', 'Clay', 'pH', 'C/N']
    for k in keys : 
        sigma_t[k] = abs(err_rel*mean_t[k])
    sigma_t['CaCO3'] =  abs(np.random.normal(0,2.5,t)) #in Qualiagro, CaCO3=0, so we create a small random error
    sigma_t['I'] = abs(err_rel_I*mean_t['I'])
    sigma_t['I_CF'] =  err_rel_I* mean_t['I_CF'] # represent carbon inputs from Carbon farming scenario    
    sigma_t['DI'] = np.sqrt(sigma_t['I_CF']**2 + sigma_t['I']**2)

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
    LUT['DI'] = LUT['I_CF'] - LUT['I']

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

#%% AMG model in the context of computing the difference of SOC stock between two management scenarios under a dynamic specific baseline

### To simplify the notation in the code, we call this difference of SOC stock between two management practices scenarios "carbon credit"

def carbon_credit_AMG(input_LUT, stress_LUT,t,n):
    LUT={}
    D = np.zeros((t,n))
    DC_inp_hum = input_LUT['DI']
    
    for i in range(t-1) : 
        dD = DC_inp_hum[i,:] - stress_LUT['mineralisation_rate'][i,:]*D[i,:]
        D[i+1,:] = dD + D[i,:]
    LUT['D'] = D
    return LUT


#%% Compute everything

## To store the outputs of the Monte Carlo simulations
dir_path = "../outputs/LUT_MC/carbon_credit/"
# Create the repertory if needed
dir_path = Path(dir_path)
dir_path.mkdir(parents=True, exist_ok=True)


## To store the sigma's vector that will be used for the analytical uncertainty propagation methods as inputs
dir_path_sigma_t = "../inputs/sigma_t/carbon_credit/"
# Create the repertory if needed
dir_path_sigma_t = Path(dir_path_sigma_t)
dir_path_sigma_t.mkdir(parents=True, exist_ok=True)


"""
With a dynamic specific baseline design, the initial conditions (initial SOC stock $C_0$ and stable/total ratio $r$) don't contribute anymore in the expression of the expected value and uncertainty of the SOC stock difference between two management practices scenario. Therefore, the error input scenarios considered only errors scenarios for carbon inputs and mineralisation rate.

The name of the objects below follow this convention : 'term1_term2_sigma_t_treatment' 

(i) The first term 'term1' corresponds to the error scenarios concerning  and input variables to compute intermediate variables I and k (carbon inputs and mineralisation rate).

- ACEO : when the scenario concerning carbon inputs uncertainty is Field-scale scenario ('ACEO' for using AgriCarbon-EO error), and 
- regstatk10 : when the scenario concerning carbon inputs uncertainty is Large-scale scenario ('regstat' for using regional statistics error), and 

(ii) The second term 'term2' corresponds to the error scenarios concerning pedoclimatic input variables to compute the intermediate variable k i.e., the mineralisation rate. 

- k10 : when the error scenario for computing the mineralisation rate uncertainty is Field-scale scenario (k10 as in the field-scale scenario we assume a 10% relative error).
- kLS : when the error scenario for computing the mineralisation rate uncertainty is Large-scale scenario (kLS for Large-Scale error).

(iii) treatment : either 'LEG' or 'PRO'

"""


# first generate the sigma vectors with dimension time t

ACEO_k10_sigma_t_LEG = input_err_I_ACEO_k10(mean_t_LEG, err_rel=0.1, err_rel_I=err_rel_I_DAM)
ACEO_k10_sigma_t_PRO = input_err_I_ACEO_k10(mean_t_PRO, err_rel=0.1, err_rel_I=err_rel_I_DAM)

ACEO_kLS_sigma_t_LEG = input_err_I_ACEO_kLS(mean_t_LEG, sigma_LS, err_rel_I=err_rel_I_DAM)
ACEO_kLS_sigma_t_PRO = input_err_I_ACEO_kLS(mean_t_PRO, sigma_LS, err_rel_I=err_rel_I_DAM)

regstat_kLS_sigma_t_LEG = input_err_I_regstat_kLS(mean_t_LEG, sigma_LS, err_rel_I=err_rel_I_Yp)
regstat_kLS_sigma_t_PRO = input_err_I_regstat_kLS(mean_t_PRO, sigma_LS, err_rel_I=err_rel_I_Yp)

regstat_k10_sigma_t_LEG = input_err_I_regstat_k10(mean_t_LEG, err_rel=0.1, err_rel_I=err_rel_I_Yp)
regstat_k10_sigma_t_PRO = input_err_I_regstat_k10(mean_t_PRO, err_rel=0.1, err_rel_I=err_rel_I_Yp)

np.save(dir_path_sigma_t + 'ACEO_k10_sigma_t_LEG.npy', ACEO_k10_sigma_t_LEG)
np.save(dir_path_sigma_t + 'ACEO_k10_sigma_t_PRO.npy', ACEO_k10_sigma_t_PRO)

np.save(dir_path_sigma_t + 'ACEO_kLS_sigma_t_LEG.npy', ACEO_kLS_sigma_t_LEG)
np.save(dir_path_sigma_t + 'ACEO_kLS_sigma_t_PRO.npy', ACEO_kLS_sigma_t_PRO)

np.save(dir_path_sigma_t + 'regstat_kLS_sigma_t_LEG.npy', regstat_kLS_sigma_t_LEG)
np.save(dir_path_sigma_t + 'regstat_kLS_sigma_t_PRO.npy', regstat_kLS_sigma_t_PRO)

np.save(dir_path_sigma_t + 'regstat_k10_sigma_t_LEG.npy', regstat_k10_sigma_t_LEG)
np.save(dir_path_sigma_t + 'regstat_k10_sigma_t_PRO.npy', regstat_k10_sigma_t_PRO)

# generate LUT inputs for AMG with numerical sampling
n = 100000 # sample size of the numerical sampling
keys = ['T', 'H', 'Clay', 'CaCO3', 'pH', 'C/N', 'I', 'I_CF']


ACEO_k10_LUT_LEG = generate_LUT(mean_t_LEG, ACEO_k10_sigma_t_LEG, keys, t, n)
ACEO_k10_LUT_PRO = generate_LUT(mean_t_PRO, ACEO_k10_sigma_t_PRO, keys, t, n)

ACEO_kLS_LUT_LEG = generate_LUT(mean_t_LEG, ACEO_kLS_sigma_t_LEG, keys, t, n)
ACEO_kLS_LUT_PRO = generate_LUT(mean_t_PRO, ACEO_kLS_sigma_t_PRO, keys, t, n)

regstat_kLS_LUT_LEG = generate_LUT(mean_t_LEG, regstat_kLS_sigma_t_LEG, keys, t, n)
regstat_kLS_LUT_PRO = generate_LUT(mean_t_PRO, regstat_kLS_sigma_t_PRO, keys, t, n)

regstat_k10_LUT_LEG = generate_LUT(mean_t_LEG, regstat_k10_sigma_t_LEG, keys, t, n)
regstat_k10_LUT_PRO = generate_LUT(mean_t_PRO, regstat_k10_sigma_t_PRO, keys, t, n)

np.save(dir_path + f'ACEO_k10_LUT_LEG_n_{n}.npy', ACEO_k10_LUT_LEG)
np.save(dir_path + f'ACEO_k10_LUT_PRO_n_{n}.npy', ACEO_k10_LUT_PRO)

np.save(dir_path + f'ACEO_kLS_LUT_LEG_n_{n}.npy', ACEO_kLS_LUT_LEG)
np.save(dir_path + f'ACEO_kLS_LUT_PRO_n_{n}.npy', ACEO_kLS_LUT_PRO)

np.save(dir_path + f'regstat_kLS_LUT_LEG_n_{n}.npy', regstat_kLS_LUT_LEG)
np.save(dir_path + f'regstat_kLS_LUT_PRO_n_{n}.npy', regstat_kLS_LUT_PRO)

np.save(dir_path + f'regstat_k10_LUT_LEG_n_{n}.npy', regstat_k10_LUT_LEG)
np.save(dir_path + f'regstat_k10_LUT_PRO_n_{n}.npy', regstat_k10_LUT_PRO)

# apply stress k

stress_ACEO_k10_LUT_LEG = mineralisation_process(ACEO_k10_LUT_LEG)
stress_ACEO_k10_LUT_PRO = mineralisation_process(ACEO_k10_LUT_PRO)

stress_ACEO_kLS_LUT_LEG = mineralisation_process(ACEO_kLS_LUT_LEG)
stress_ACEO_kLS_LUT_PRO = mineralisation_process(ACEO_kLS_LUT_PRO)

stress_regstat_kLS_LUT_LEG = mineralisation_process(regstat_kLS_LUT_LEG)
stress_regstat_kLS_LUT_PRO = mineralisation_process(regstat_kLS_LUT_PRO)

stress_regstat_k10_LUT_LEG = mineralisation_process(regstat_k10_LUT_LEG)
stress_regstat_k10_LUT_PRO = mineralisation_process(regstat_k10_LUT_PRO)

np.save(dir_path + f'stress_ACEO_k10_LUT_LEG_n_{n}.npy', stress_ACEO_k10_LUT_LEG)
np.save(dir_path + f'stress_ACEO_k10_LUT_PRO_n_{n}.npy', stress_ACEO_k10_LUT_PRO)

np.save(dir_path + f'stress_ACEO_kLS_LUT_LEG_n_{n}.npy', stress_ACEO_kLS_LUT_LEG)
np.save(dir_path + f'stress_ACEO_kLS_LUT_PRO_n_{n}.npy', stress_ACEO_kLS_LUT_PRO)

np.save(dir_path + f'stress_regstat_kLS_LUT_LEG_n_{n}.npy', stress_regstat_kLS_LUT_LEG)
np.save(dir_path + f'stress_regstat_kLS_LUT_PRO_n_{n}.npy', stress_regstat_kLS_LUT_PRO)

np.save(dir_path + f'stress_regstat_k10_LUT_LEG_n_{n}.npy', stress_regstat_k10_LUT_LEG)
np.save(dir_path + f'stress_regstat_k10_LUT_PRO_n_{n}.npy', stress_regstat_k10_LUT_PRO)


# apply AMG model in the context of the estimation of a SOC stock difference between two management practices scenario under a dynamic specific baseline design

AMG_ACEO_k10_LUT_LEG = carbon_credit_AMG(ACEO_k10_LUT_LEG, stress_ACEO_k10_LUT_LEG,t,n)
AMG_ACEO_k10_LUT_PRO = carbon_credit_AMG(ACEO_k10_LUT_PRO, stress_ACEO_k10_LUT_PRO,t,n)

AMG_ACEO_kLS_LUT_LEG = carbon_credit_AMG(ACEO_kLS_LUT_LEG, stress_ACEO_kLS_LUT_LEG,t,n)
AMG_ACEO_kLS_LUT_PRO = carbon_credit_AMG(ACEO_kLS_LUT_PRO, stress_ACEO_kLS_LUT_PRO,t,n)

AMG_regstat_kLS_LUT_LEG = carbon_credit_AMG(regstat_kLS_LUT_LEG, stress_regstat_kLS_LUT_LEG,t,n)
AMG_regstat_kLS_LUT_PRO = carbon_credit_AMG(regstat_kLS_LUT_PRO, stress_regstat_kLS_LUT_PRO,t,n)

AMG_regstat_k10_LUT_LEG = carbon_credit_AMG(regstat_k10_LUT_LEG, stress_regstat_k10_LUT_LEG,t,n)
AMG_regstat_k10_LUT_PRO = carbon_credit_AMG(regstat_k10_LUT_PRO, stress_regstat_k10_LUT_PRO,t,n)

np.save(dir_path + f'AMG_ACEO_k10_LUT_LEG_n_{n}.npy', AMG_ACEO_k10_LUT_LEG)
np.save(dir_path + f'AMG_ACEO_k10_LUT_PRO_n_{n}.npy', AMG_ACEO_k10_LUT_PRO)

np.save(dir_path + f'AMG_ACEO_kLS_LUT_LEG_n_{n}.npy', AMG_ACEO_kLS_LUT_LEG)
np.save(dir_path + f'AMG_ACEO_kLS_LUT_PRO_n_{n}.npy', AMG_ACEO_kLS_LUT_PRO)

np.save(dir_path + f'AMG_regstat_kLS_LUT_LEG_n_{n}.npy', AMG_regstat_kLS_LUT_LEG)
np.save(dir_path + f'AMG_regstat_kLS_LUT_PRO_n_{n}.npy', AMG_regstat_kLS_LUT_PRO)

np.save(dir_path + f'AMG_regstat_k10_LUT_LEG_n_{n}.npy', AMG_regstat_k10_LUT_LEG)
np.save(dir_path + f'AMG_regstat_k10_LUT_PRO_n_{n}.npy', AMG_regstat_k10_LUT_PRO)


