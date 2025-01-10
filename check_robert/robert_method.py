import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from numpy.random import normal, random, multivariate_normal
from scipy.interpolate import interp1d


data = np.loadtxt('real_data.txt')
cov = np.loadtxt('cov_matrix.txt')
theory = np.loadtxt('bestfit_theory.txt')
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
for i in range(100):
    sample = np.loadtxt(f'{i}' + '.txt')
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

# Create a histogram of the chi^2 values with more bins
counts, bins, _ = plt.hist(chi2, bins=8, density=True, edgecolor='black', color='skyblue', label='mocks')
# The center of each bin for fitting
bin_centers = 0.5 * (bins[1:] + bins[:-1])

# Add a vertical dashed line for the chi^2 best-fit
plt.axvline(chi2_observed , color='darkblue', linestyle='--', linewidth=2, label=r'KiDS$\times$BOSS data')

# Make x and y ticks thicker
plt.tick_params(axis='both', which='major', width=2, length=6)  # For both x and y axes
# Increase the thickness of the tick labels
plt.tick_params(axis='both', which='major', labelsize=30)

# Add labels, legend, and show the plot
#plt.xlim(300,600)
plt.xlabel(r'$\chi^2$', fontsize=35)
plt.ylabel(r'$P(\chi^2)$', fontsize=35)
plt.legend(fontsize=30)
plt.show()