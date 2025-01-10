import matplotlib.pyplot as plt
import numpy as np
import os 

ellbin = np.logspace(np.log10(76), np.log10(1500), 8+1)
#data vector
data_vec_path = os.path.join('/Users/matteograsso/Desktop/DATA/kids1000_pcl/DATA/', 'PKWL-EE-DATAVEC.dat')
data_vec = np.loadtxt(data_vec_path)
data = np.array_split(data_vec,15)
#best-fit
#temp = np.loadtxt('best_fit_unbinned.dat') #mean=my Horndeski best-fit
#best_fit = np.array_split(temp,15)

#loading mocks data
mocks = []
for i in range(0,10):
    mock = np.loadtxt('/Users/matteograsso/Desktop/DATA/mock_data/' +  f'PKWL-EE-DATAVEC_{i}.dat')
    mock_split = np.array_split(mock,15)
    mocks.append(mock_split)

#data
#data_binned = np.array_split(data_vec,15)

# Assuming mocks is a list of 10 nested lists (each with 15 sublists, each sublist with 8 points)
# best_fit is a list with 15 sublists (each with 8 points)
# lll is a list of multipole values (length = 8)


# Create a figure with subplots (5 rows, 3 columns to accommodate 15 bins)
fig, axes = plt.subplots(5, 3, figsize=(8, 8))
axes = axes.flatten()  # Flatten the axes array for easy indexing
# Loop through bins and plot each bin in a separate panel
for bin_idx in range(15):
    ax = axes[bin_idx]
    
    # Plot the best-fit for the current bin
    #ax.plot(ellbin[1:], best_fit[bin_idx], label=f'Best-fit', color='black', linestyle = 'dotted', linewidth=2)
    ax.plot(ellbin[1:], data[bin_idx], 'o', label=f'data', color='red', markersize=6)

    
     # Plot the mock data for the current bin (all 10 mocks)
    for j in range(10):
        ax.plot(ellbin[1:], mocks[j][bin_idx], 'o', alpha=0.4, markersize=4)
    
    ax.set_title(f'Bin {bin_idx + 1}')
    ax.set_xlabel('Multipoles (l)')
    ax.set_ylabel('Power Spectrum')

    
# Adjust layout to avoid overlap
plt.legend()
plt.tight_layout()
plt.show()
