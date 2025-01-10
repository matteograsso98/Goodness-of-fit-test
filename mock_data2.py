#the difference between this script and mock_data.py is that here the MEAN is set equal to the REAL DATA VECTOR 
#instead of the best-fit theory line
import numpy as np
from numpy.random import normal, random, multivariate_normal
import healpy as hp

data_dir = '/Users/matteograsso/Desktop/DATA/kids1000_pcl/DATA'
cov = np.loadtxt(data_dir + '/PKWL-Covariance-Matrix.dat') #load covariance
data = np.loadtxt(data_dir + '/PKWL-EE-DATAVEC.dat') #load data; 8 data points for 15 z-bins = 120 points
pixWin = (np.power(hp.pixwin(1024, pol=True, lmax=3070), 2)[0]) #pixel window function
lll = range(2, 3071) #2,3,...,3070
ellbin = np.logspace(np.log10(76), np.log10(1500), 8+1)
mocksVarGold = cov.diagonal().reshape(15,8) #error bars from the covariance


N=10 #number of mocks
for i in range(N):
    fake_data = np.random.multivariate_normal(data, cov).T
    np.savetxt('/Users/matteograsso/Desktop/DATA/mock_data/PKWL-EE-DATAVEC_' + f'{i}' + '.dat', fake_data)