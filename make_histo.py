import os
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
from scipy.optimize import curve_fit

"""
# Directory and file pattern
#folder_path = '/Users/matteograsso/Desktop/'
#file_prefix = 'KiDSxBOSS_mock_chi2_'
# List to store the chi^2 values
chi_squared_values = []

# Loop over files with the _i suffix from 0 to 49
for i in range(15):
    file_path = os.path.join(folder_path, f"{file_prefix}{i}.txt")
    
    # Open and read the first line of the file
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
        
        # Extract the chi^2 value using a regular expression
        match = re.search(r"# minimized \\chi\^2 = ([\d.]+)", first_line)
        if match:
            chi_squared_value = float(match.group(1))  # Convert to float
            chi_squared_values.append(chi_squared_value)
"""

# Load chi^2 values from the provided file
chi_squared_values = np.loadtxt('/Users/matteograsso/Desktop/cosmo_mocks/chi_squared_values.txt')

# Define the function for the chi-squared distribution PDF
def chi2_pdf(x, df, loc):
    return chi2.pdf(x, df, loc=loc)

"""
# Create the histogram data (but don't plot the histogram)
counts, bins = np.histogram(chi_squared_values, bins=12, density=True)
# The center of each bin for plotting and fitting
bin_centers = 0.5 * (bins[1:] + bins[:-1])
# Plot the bin centers as points (scatter plot)
plt.scatter(bin_centers, counts, color='blue', label='mocks')
"""

# Create a histogram of the chi^2 values with more bins
counts, bins, _ = plt.hist(chi_squared_values, bins=8, density=True, edgecolor='black', color='skyblue', label='mocks')
# The center of each bin for fitting
bin_centers = 0.5 * (bins[1:] + bins[:-1])

# Initial guess for degrees of freedom (df) and location (loc)
initial_guess = [105, 300]  # loc is set to the minimum chi2 value, df is an initial guess
# Fit the histogram using the chi-squared distribution
params, cov_matrix = curve_fit(chi2_pdf, bin_centers, counts, p0=initial_guess)

# Extract fitted parameters
df_fit, loc_fit = params

# Generate the x values for plotting the fitted chi^2 distribution
x = np.linspace(min(chi_squared_values), max(chi_squared_values), 1000)
pdf_fit = chi2_pdf(x, df_fit, loc_fit)


# Plot the fitted chi^2 distribution
plt.plot(x, pdf_fit, color='blue')#,label=f'$\chi^2$ (df={df_fit:.2f}, loc={loc_fit:.2f})')

# Add a vertical dashed line for the chi^2 best-fit
chi2_best_fit = 409  # Replace with your actual chi^2 best-fit value
plt.axvline(chi2_best_fit, color='darkblue', linestyle='--', linewidth=2, label=r'KiDS$\times$BOSS data')

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