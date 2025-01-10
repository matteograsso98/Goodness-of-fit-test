import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from numpy.random import normal, random, multivariate_normal
from scipy.interpolate import interp1d
import healpy as hp

data_dir = '/Users/matteograsso/Desktop/DATA/kids1000_pcl/DATA'
cov = np.loadtxt(data_dir + '/PKWL-Covariance-Matrix.dat') #load covariance
data = np.loadtxt(data_dir + '/PKWL-EE-DATAVEC.dat') #load data; 8 data points for 15 z-bins = 120 points
data_binned =  np.array_split(data,15) #split the data, 8 for each bin
pixWin = (np.power(hp.pixwin(1024, pol=True, lmax=3070), 2)[0]) #pixel window function
lll = range(2, 3071) #2,3,...,3070
ellbin = np.logspace(np.log10(76), np.log10(1500), 8+1)
mocksVarGold = cov.diagonal().reshape(15,8) #error bars from the covariance

df_data = pd.DataFrame() #data
#mean_binned_pix = [pixWin[2:]*i for i in mean_binned]
df_mean_int = pd.DataFrame() #interpolated; best-fit mean for each z-bin
df_errorbar = pd.DataFrame()

#cast data points into redshift bins called "E1-E1, E1-E2, etc."
names = []
for i in range(1,6):
    for j in range(1,6):
        if i > j:
            #axis[i,j].axis('off')
            None
        else:
            x = f'E{i}-E{j}'
            names.append(x)

#######################
#Now we build our mean 
#######################

#mean = np.loadtxt('/Users/matteograsso/Desktop/MSc_thesis/Paper_notebook/Paper_plots/PCL_screened.dat') #mean = Hor best-fit
mean = np.loadtxt('/Users/matteograsso/Desktop/MSc_thesis/Paper_notebook/Paper_plots/pcl_hor_k1k.dat')
#mean = np.loadtxt('/Users/matteograsso/Desktop/MSc_thesis/Paper_notebook/Paper_plots/pcl_hor_k1k.dat')
mean_binned = np.array_split(mean,15)  #len is 3069 for each of the 15 elems
mean_binned_pix = [i*pixWin[2:] for i in mean_binned]
df_mean_int = pd.DataFrame() #interpolated -- best-fit mean for each z-bin

for i in range(len(mean_binned)):
    p = interp1d(lll[:], mean_binned_pix[i], bounds_error=False, kind = 'cubic')
    df_mean_int[names[i]] = p(ellbin[1:]) #mean_binned[i]

mean_vector = df_mean_int.values.flatten(order='F') # This flattens column by column
np.savetxt('mean_unbinned_pix.dat', mean_vector)
