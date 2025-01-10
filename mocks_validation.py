#in this script we recast my best-fit (theory) vector into the data format (8 pts for each of the 15 redshift bins). Then I check if the interpolation 
#works by plotting the interpolation vs. the non-interpolated theory vector. We also plot the data points to show that there's no huge difference between the 
#best-fit "fake" data pts and the real data points. MY MEAN IS (ellbin[1:], df_mean_int[name])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from numpy.random import normal, random, multivariate_normal
from scipy.interpolate import interp1d
import healpy as hp

data_dir = '/Users/matteograsso/Desktop/DATA/kids1000_pcl/DATA'
cov = np.loadtxt(data_dir + '/PKWL-Covariance-Matrix.dat')
data = np.loadtxt(data_dir + '/PKWL-EE-DATAVEC.dat') #8 data points *15 z-bins = 120 points
data_binned =  np.array_split(data,15) 
pixWin = (np.power(hp.pixwin(1024, pol=True, lmax=3070), 2)[0]) #pixel window function:
mean = np.loadtxt('/Users/matteograsso/Desktop/MSc_thesis/Paper_notebook/Paper_plots/PCL_screened.dat') #mean = Hor best-fit; 
mean_binned = np.array_split(mean,15)  #len is 3069 for each of the 15 elems
lll = range(2, 3071) #2,3,...,3070
ellbin = np.logspace(np.log10(76), np.log10(1500), 8+1)

# getting the error-bars from the covariance:
mocksVarGold = cov.diagonal().reshape(15,8)

df_data = pd.DataFrame() #data
df_hor_bar_screened = pd.DataFrame() #Horndeski, baryonic feedback, screened
#mean_binned_pix = [pixWin[2:]*i for i in mean_binned]
df_mean_int = pd.DataFrame() #interpolated -- best-fit mean for each z-bin
df_mean = pd.DataFrame() #not interpolated 
df_errorbar = pd.DataFrame()
names = []

for i in range(1,6):
    for j in range(1,6):
        if i > j:
            #axis[i,j].axis('off')
            None
        else:
            x = f'E{i}-E{j}'
            names.append(x)

for i in range(len(data_binned)):
    df_data[names[i]] = data_binned[i]*1e7
    df_errorbar[names[i]] = np.sqrt(mocksVarGold[i])*1e7
for i in range(len(mean_binned)):
    p = interp1d(lll[:], mean_binned[i], bounds_error=False, kind = 'cubic')
    df_mean_int[names[i]] = p(ellbin[1:])*1e7 #mean_binned[i]
    df_mean[names[i]] = mean_binned[i]*1e7

#plot for checking
colours_bins = ['#FF595E', '#FFCA3A', '#8AC926', '#1982C4', '#6A4C93']
plt.rcParams.update({'font.size': 18})
fig, axis = plt.subplots(figsize=(15,15), sharex='col', sharey=True, ncols=6, nrows=6, gridspec_kw={'wspace': 0, 'hspace': 0})
axis[0,0].axis('off')
plt.setp(axis[0,1].get_xticklabels(), visible=True)

for i in range(1,6):
    axis[i,0].axis('off')
    for j in range(1,6):
        if i > j:
            axis[i,j].axis('off')
            None
        else:
            name = f'E{i}-E{j}'
            title = f'$E_{i}$-$E_{j}$'
            
            if (i==5) and (j==5):
                axis[i-1,j].text(.5, .05, f"{title}", fontsize=18, 
                             horizontalalignment='center',
                             transform=axis[i-1,j].transAxes)
                
            else:
                axis[i-1,j].text(.5, .85, f"{title}", fontsize=18, 
                             horizontalalignment='center',
                             transform=axis[i-1,j].transAxes)
            
            axis[i,j].set_xscale('log')
            axis[i,j].set_xlim(50,2000)
            plt.tick_params(axis='x', which='minor')
            axis[i-1,j].set_xlabel(r"$\ell$")
            
            if i == j: 
                axis[i,j].axis('off')
            ######################### HERE WE PLOT #########################
            #not-interpolated
            axis[i-1,j].plot(lll[:], df_mean[name]*pixWin[2:], label='best-fit', linestyle = '--', color='blue', linewidth=2.)
            #data
            axis[i-1,j].errorbar(ellbin[1:], df_data[name], yerr=df_errorbar[name], 
                         label='PCL E-Mode', fmt="o", capsize=2, elinewidth=2, color = 'k')
            #interpolation
            axis[i-1,j].plot(ellbin[1:], df_mean_int[name], 'o', markersize=4.5, label='interpolation', color = 'red')
            # zero lines:
            axis[i-1,j].axhline(0, ls='--', color='k')
            axis[i-1,j].set_xscale('log')
            axis[i-1,j].set_xlim(50,2000)
            axis[j,i-1].set_ylim(-1.8e-6*1e7,1.8e-6*1e7)
            
            # solving the xticks issue!
            x_major = matplotlib.ticker.LogLocator(base = 10.0, numticks = 5)
            axis[j,i-1].xaxis.set_major_locator(x_major)
            axis[i-1,j].xaxis.set_major_locator(x_major)
            x_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
            axis[j,i-1].xaxis.set_minor_locator(x_minor)
            axis[j,i-1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            axis[i,j].xaxis.set_minor_locator(x_minor)
            axis[i,j].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            axis[i-1,j].xaxis.set_tick_params(which='both', labelbottom=True)
            
            axis[i,5].yaxis.set_tick_params(which='both', labelright='on', labelleft='off')
            axis[i,5].yaxis.tick_right()        
            
handlesE, labelsE = axis[0,1].get_legend_handles_labels()
handlesB, labelsB = axis[1,0].get_legend_handles_labels()
fig.legend(handlesE + handlesB, labelsE + labelsB, loc='lower center', ncol=5, bbox_transform = plt.gcf().transFigure, bbox_to_anchor = (.1,.15,1,1))

axis[5,0].set_xlabel("$\ell$")
axis[5,1].set_xlabel("$\ell$")
axis[5,2].set_xlabel("$\ell$")
axis[5,3].set_xlabel("$\ell$")
axis[5,4].set_xlabel("$\ell$")

axis[0,5].yaxis.set_tick_params(which='both', labelright='on', labelleft='off')
axis[0,5].yaxis.tick_right()


#fig.text(0.5, -0.14, r"$\ell$", ha='center')
fig.text(0.95, 0.6, r"$[\ell(\ell+1)/2\pi]\, C_{\ell} \,\times 10^{7}$"  , va='center', rotation='vertical')

#             if i == 4:
#                 axis[i-1,j].legend(loc=0, ncol=1, bbox_to_anchor=(-3.5,1))

plt.show()
