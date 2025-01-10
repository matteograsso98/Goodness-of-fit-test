import numpy as np
import matplotlib.pyplot as plt

data_dir = '/Users/matteograsso/Desktop/DATA/kids1000_pcl/DATA'
# Inputs
d_obs = np.array(np.loadtxt(data_dir + '/PKWL-EE-DATAVEC.dat'))  # Observed data vector, 8 pts for 15 z-bins = 120 points
t_best_fit = np.array(np.loadtxt('mean_unbinned_pix.dat'))  # Best-fit theory vector
cov_matrix = np.array(np.loadtxt(data_dir + '/PKWL-Covariance-Matrix.dat') )  # Covariance matrix
num_simulations = 100000  # Number of simulations

# Invert the covariance matrix
cov_inv = np.linalg.inv(cov_matrix)

# Function to compute chi^2
def compute_chi2(data, theory, cov_inv):
    delta = data - theory
    return delta.T @ cov_inv @ delta

# Compute chi^2 for the observed data
chi2_obs = compute_chi2(d_obs, t_best_fit, cov_inv)

# Generate synthetic data and compute chi^2 for each
simulated_chi2_values = []
for _ in range(num_simulations):
    # Generate random noise from the multivariate Gaussian
    noise = np.random.multivariate_normal(mean=np.zeros_like(d_obs), cov=cov_matrix)
    
    # Create synthetic data
    d_sim = t_best_fit + noise
    
    # Compute chi^2 for this synthetic data realization
    chi2_sim = compute_chi2(d_sim, t_best_fit, cov_inv)
    
    # Store the result
    simulated_chi2_values.append(chi2_sim)

# Convert to a numpy array for easier handling
simulated_chi2_values = np.array(simulated_chi2_values)

# Compute the p-value
p_value = np.sum(simulated_chi2_values > chi2_obs) / num_simulations
print(f"p-value: {p_value}")


# Optional: Plot histogram of chi^2 values and mark the observed chi^2
plt.hist(simulated_chi2_values, bins=50, alpha=0.7, label="Simulated chi^2")
plt.axvline(chi2_obs, color='r', linestyle='--', label=f"Observed chi^2 = {chi2_obs:.2f}")
plt.xlabel('chi^2')
plt.ylabel('Frequency')
plt.title('Chi^2 Distribution of Simulated Data')
plt.legend()
plt.show()


