import numpy as np
import matplotlib.pyplot as plt

def line(x,m,q):
    return m*x+q

x=np.linspace(0,50,100)
#create the "real data" 
#Add noise to the data (normally distributed)
noise = np.random.normal(0, 1, size=line(x,1,2).shape)  # noise with mean 0 and standard deviation 1
real_data = line(x,1,2) + noise
np.savetxt('real_data.txt', real_data) 

# Fit a line to the noisy data
best_fit_params = np.polyfit(x, real_data, 1)  # Degree 1 for a linear fit
m_fit, q_fit = best_fit_params  # slope and intercept from the fit
# Generate the best-fit line using the fit parameters
y_fit = m_fit * x + q_fit
np.savetxt('bestfit_theory.txt', y_fit)

#create N mock realizations 
for i in range(100):
    n = np.random.normal(0, 1, size=line(x,1,2).shape)
    mocks= y_fit + n
    np.savetxt(f'{i}' + '.txt', mocks)

# Create the covariance matrix for the noise (diagonal with 1's on the diagonal)
n_points = len(x)  # number of data points
cov_matrix = np.eye(n_points)  # Identity matrix with ones on the diagonal, representing uncorrelated noise
np.savetxt('cov_matrix.txt', cov_matrix)  # Display the covariance matrix

"""
# Plot the noisy data and the original straight line
plt.figure(figsize=(8, 5))
plt.plot(x, line(x,1,2), label="True line: y=mx+q", color='blue', linestyle='--')
plt.plot(x, y_fit, label=f"Best-fit line: y={m_fit:.2f}x+{q_fit:.2f}", color='green', linestyle='--')
#plt.title("Best-Fit Line to Noisy Data")
plt.scatter(x, fake_data, label="Noisy data", color='red', s=10)
plt.title("Fake Data: Straight Line with Noise")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
m_fit, q_fit
"""
