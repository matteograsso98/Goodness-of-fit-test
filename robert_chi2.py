import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from numpy.random import normal, random, multivariate_normal
from scipy.interpolate import interp1d


data = np.loadtxt('/Users/matteograsso/Desktop/DATA/kids1000_pcl/DATA/PKWL-EE-DATAVEC.dat')
cov = np.loadtxt('/Users/matteograsso/Desktop/DATA/kids1000_pcl/DATA/PKWL-Covariance-Matrix.dat')
theory = np.loadtxt('mean_unbinned_pix.dat')
# Compute the inverse of the covariance matrix - needed for chi2 calculation
cov_inv = np.linalg.inv(cov)
chi2=[]

def calculate_p_value(chi2_samples, chi2_data):
    # Calculate the number of chi-squared values greater than the observed chi2_data
    num_greater = np.sum(chi2_samples > chi2_data)
    
    # Calculate the p-value as the fraction of chi2_samples greater than chi2_data
    p_value = num_greater / len(chi2_samples)
    
    return p_value

# chi2 = (samples - theory at best fit point)^T covariance^{-1} (samples - theory at best fit point)
for i in range(1000):
    sample = np.loadtxt('robert_chi2/' + f'{i}' + '.dat')
    # Calculate the difference between the samples and the theory at the best-fit point
    delta = sample - theory 
    # Calculate the chi-squared value
    chi2_value = np.dot(delta.T, np.dot(cov_inv, delta))
    chi2.append(chi2_value)

delta_observed = data - theory 
chi2_observed = np.dot(delta_observed.T, np.dot(cov_inv, delta_observed))

#print(chi2_observed)
#print(chi2)

print(calculate_p_value(chi2,chi2_observed))