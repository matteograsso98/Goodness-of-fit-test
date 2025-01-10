import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import classy
import os
import matplotlib
import healpy as hp
from scipy import interpolate as itp
from scipy import stats

#I need this function to load the mixing matrix 
def __one_dim_index(Bin1, Bin2):
    """
    This function is used to convert 2D sums over the two indices (Bin1, Bin2)
    of an N*N symmetric matrix into 1D sums over one index with N(N+1)/2
    possible values.
    """
    if Bin1 <= Bin2:
        return Bin2 + nzbins * Bin1 - (Bin1 * (Bin1 + 1)) // 2
    else:
        return Bin1 + nzbins * Bin2 - (Bin2 * (Bin2 + 1)) // 2

#useful quantities
nzbins = 5
nbins = 8
nzcorrs = nzbins * (nzbins + 1) // 2
nl = 3069
lll = range(2, 3071)
ells = np.logspace(np.log10(76), np.log10(1500), 50)
ellbin = np.logspace(np.log10(76), np.log10(1500), 8+1)
BIN_EDGES = ellbin
# ell-bin edges and pixel window function:
pixWin = (np.power(hp.pixwin(1024, pol=True, lmax=3070), 2)[0])
pixWinB = (np.power(hp.pixwin(1024, pol=True, lmax=2999), 2)[0])
ellPclsBin_edges = np.logspace(np.log10(76), np.log10(1500), 9)

#loading data and theory vectors (theory vectors are PCls already; don't need to take the product with the mixing matrix)
#data vector
data_vec_path = os.path.join('/Users/matteograsso/Desktop/DATA/kids1000_pcl/DATA/', 'PKWL-EE-DATAVEC.dat') 
data_vec = np.loadtxt(data_vec_path)
#covariance matrix
covmat_path = os.path.join('/Users/matteograsso/Desktop/DATA/kids1000_pcl/DATA/', 'PKWL-Covariance-Matrix.dat')
covmat = np.loadtxt(covmat_path)
pixelwind_path = os.path.join('/Users/matteograsso/Desktop/DATA/kids1000_pcl/DATA/', 'PIXEL_WINDOW_NSIDE1024_EMODE.dat')
pixelwind = np.loadtxt(pixelwind_path)
#read the mixing matrix
mixmat = np.zeros((nzcorrs, 3069, 3069))
for Bin1 in range(nzbins):
    for Bin2 in range(Bin1, nzbins):
        indexcorr = __one_dim_index(Bin1,Bin2)
        mixmat_file_path = os.path.join('/Users/matteograsso/Desktop/DATA/kids1000_pcl/DATA/', 'MIXING_MATRIX/z{0:}z{1:}.npy'.format(Bin1+1, Bin2+1))
        mixmat[indexcorr] = np.load(mixmat_file_path)

#theory vectors
home = '/Users/matteograsso/Desktop/MSc_thesis/Paper_notebook/Paper_plots' 
#hor_massive_nu = np.loadtxt('/Users/matteograsso/Desktop/pcl_massive_neutrinos.dat')
hor_k1k = np.loadtxt(home + '/pcl_hor_k1k.dat')
lcdm_planck = np.loadtxt(home + '/Pcl_planck_bf.dat')
lcdm_k1k = np.loadtxt(home + '/pcl_best_k1k.dat')
hor_nobar_screened = np.loadtxt(home + '/PCL_nobar_screened.dat')
hor_nobar_unscreened = np.loadtxt(home + '/PCL_nobar_unscreened.dat')
hor_bar_screened = np.loadtxt(home + '/PCL_screened.dat')
hor_bar_unscreened  = np.loadtxt(home + '/PCL_unscreened.dat')
boss_hor = np.loadtxt(home + '/pcl_baorsd_bf.dat')
planck_hor = np.loadtxt(home + '/pcl_cmb_bf.dat')
kids_only_lcdm = np.loadtxt('/Users/matteograsso/Desktop/KiDS_LCDM_bf.dat') #Arthur best-fit first column of the table in his paper
kidsxboss_lcdm = np.loadtxt('/Users/matteograsso/Desktop/KiDSxBOSS_LCDM_bf.dat') #Arthur best-fit first column of the table in his paper

#split the data (120 pts) in 15 bins --> 8 pts for each bin 
data_binned = np.array_split(data_vec,15)
hor_k1k_binned = np.array_split(hor_k1k, 15) #this should be equal to hor_bar_screened
hor_nobar_unscreened_binned = np.array_split(hor_nobar_unscreened,15)
hor_nobar_screened_binned = np.array_split(hor_nobar_screened,15)
hor_bar_unscreened_binned = np.array_split(hor_bar_unscreened,15)
hor_bar_screened_binned = np.array_split(hor_bar_screened,15)
lcdm_planck_binned = np.array_split(lcdm_planck, 15) #planck LCDM cosmology
lcdm_k1k_binned = np.array_split(lcdm_k1k, 15)
boss_hor_binned = np.array_split(boss_hor,15) #boss
planck_hor_binned = np.array_split(planck_hor,15) #planck Horndeski
kids_only_lcdm_binned = np.array_split(kids_only_lcdm,15)
kidsxboss_lcdm_binned = np.array_split(kidsxboss_lcdm,15)
# getting the error-bars from the covariance:
mocksVarGold = covmat.diagonal().reshape(15,8)

#creating the data frames 
names = []
for i in range(1,6):
    for j in range(1,6):
        if i > j:
            #axis[i,j].axis('off')
            None
        else:
            x = f'E{i}-E{j}' 
            names.append(x) 
            
df_data = pd.DataFrame() #data
df_errorbar = pd.DataFrame() #add errorbars to data
df_hor_k1k = pd.DataFrame() #Horndeski
df_lcdm_planck= pd.DataFrame() #planck LCDM
df_lcdm_planck_again = pd.DataFrame() 
df_lcdm_k1k = pd.DataFrame() #KiDSxBOSS LCDM
df_boss_hor = pd.DataFrame() #BOSS Horndeski
df_planck_hor = pd.DataFrame() #planck Horndeski
df_hor_nobar_screened = pd.DataFrame() #Horndeski, no baryonic feedback, screened
df_hor_nobar_unscreened = pd.DataFrame() #Horndeski, no baryonic feedback, UNscreened
df_hor_bar_screened = pd.DataFrame() #Horndeski, baryonic feedback, screened
df_hor_bar_unscreened = pd.DataFrame() #Horndeski, baryonic feedback, UNscreened
df_kids_only_lcdm = pd.DataFrame() #LCDM, kids only 
df_kidsxboss_lcdm = pd.DataFrame()

for i in range(len(data_binned)):
    df_data[names[i]] = data_binned[i]*1e7
    df_hor_k1k[names[i]] =  hor_k1k_binned[i]*1e7
    df_lcdm_planck[names[i]] = lcdm_planck_binned[i]*1e7
    df_lcdm_k1k[names[i]] = lcdm_k1k_binned[i]*1e7   
    df_boss_hor[names[i]] = boss_hor_binned[i]*1e7
    df_planck_hor[names[i]] = planck_hor_binned[i]*1e7
    df_hor_nobar_screened[names[i]] = hor_nobar_screened_binned[i]*1e7
    df_hor_nobar_unscreened[names[i]] = hor_nobar_unscreened_binned[i]*1e7
    df_hor_bar_screened[names[i]] = hor_bar_screened_binned[i]*1e7
    df_hor_bar_unscreened[names[i]] = hor_bar_unscreened_binned[i]*1e7
    df_kids_only_lcdm[names[i]] = kids_only_lcdm_binned[i]*1e7
    df_kidsxboss_lcdm[names[i]] = kidsxboss_lcdm_binned[i]*1e7
    df_errorbar[names[i]] = np.sqrt(mocksVarGold[i])*1e7

#plot 
colours_bins = ['#FF595E', '#FFCA3A', '#8AC926', '#1982C4', '#6A4C93', 	'#4C936A']
plt.rcParams.update({'font.size': 24})
fig, axis = plt.subplots(figsize=(15,15), sharex='col', sharey=True, ncols=2, nrows=2, gridspec_kw={'wspace': 0, 'hspace': 0})

for i in range(0,2):
    for j in range(0,2):
        if i>j: 
            axis[i,j].axis('off')
        else:    
            name = f'E{i+4}-E{j+4}'
            title = f'$E_{i+4}$-$E_{j+4}$'
            axis[i,j].text(.5, .95, f"{title}", fontsize=24, horizontalalignment='center',
            transform=axis[i,j].transAxes)
            axis[i,j].set_xscale('log')
            axis[i,j].yaxis.tick_right()
            # make some labels invisible
            axis[i,j].xaxis.set_tick_params(labelbottom=True)
            axis[i,j].yaxis.set_tick_params(labelleft=True)
            #HORNDESKI KiDS only?
            axis[i,j].plot(lll[:], df_hor_k1k[name]*pixWin[2:], 
                           label='KiDS Hor ', linestyle = '-', color='black', linewidth=2)#c=colours_bins[5])
            
            #BOSS (HORNDESKI)
            #axis[i,j].plot(lll[:], df_boss_hor[name]*pixWin[2:], 
                           #label='BAO&RSD Hor ', linestyle = '-', color='black', linewidth=2)#c=colours_bins[5])
            #PLANCK (HORNDESKI)
            #axis[i,j].plot(lll[:], df_planck_hor[name]*pixWin[2:], 
                           #label='KiDS×BAO&RSD×Planck 2018 Hor', linestyle = '-', color='dodgerblue', linewidth=2.5)#c=colours_bins[5])
            #HORNDESKI 
            #axis[i,j].plot(lll[:], df_hor_nobar_screened[name]*pixWin[2:], 
                           #label='Hor KiDS-1000 $P\mathcal{C}_l$ + BAO & RSD, no baryonic feedback + S', linestyle = '-.', c=colours_bins[2])
            #BEST-FIT: kidsxboss, Horndeski + screening 
            axis[i,j].plot(lll[:], df_hor_bar_screened[name]*pixWin[2:], 
                           label='KiDS×BAO&RSD Hor, screened', linestyle = '-', color='orange', linewidth=2.5) #c=colours_bins[3])
            #BEST-FIT without screening 
            axis[i,j].plot(lll[:], df_hor_bar_unscreened[name]*pixWin[2:], 
                           label='KiDS×BAO&RSD Hor, unscreened ', linestyle = ':', color='orange',linewidth=2.5)#c=colours_bins[4])
            #KIDS-1000 + BOSS LCDM
            axis[i,j].plot(lll[:], df_lcdm_k1k[name]*pixWin[2:], 
                           label='KiDS×BAO&RSD LCDM',  linestyle = '-', color='darkred', linewidth=2.)#c=colours_bins[0])
            #Arthur KIDS LCDM
            axis[i,j].plot(lll[:], df_kids_only_lcdm[name]*pixWin[2:], 
                           label='KiDS LCDM',  linestyle = '-', color='darkred', linewidth=2.)#c=colours_bins[0])
            #Arthur KIDSxBOSS LCDM
            axis[i,j].plot(lll[:], df_kidsxboss_lcdm[name]*pixWin[2:], 
                           label='KiDSxBOSS LCDM',  linestyle = '-', color='darkblue', linewidth=2.)#c=colours_bins[0])
            
            #PLANCK LCDM
            #axis[i,j].plot(lll[:], df_lcdm_planck[name]*pixWin[2:], label='Planck 2018 LCDM', linestyle = '-', color='limegreen', linewidth=2. )
            #DATA VECTOR
            axis[i,j].errorbar(ellbin[1:], df_data[name], yerr=df_errorbar[name], 
                         label='PCL E-Mode', fmt="o", capsize=2, elinewidth=2, color = 'k')

            axis[i,j].set_xlabel("$\ell$", fontsize=24)

            plt.legend(loc = 'center right', bbox_to_anchor=(-0.1, 0.5));
            
fig.text(0.92, 0.6, r"$[\ell(\ell+1)/2\pi]\, C_{\ell} \,\times 10^{7}$"  , va='center', rotation='vertical', fontsize= 26)
axis[0,0].set_ylim(-.5,1.3e-6*1e7)
axis[0,1].set_ylim(-.5,1.3e-6*1e7)
axis[1,0].set_ylim(-1,1.45e-6*1e7)

# make some labels invisible
axis[0,1].xaxis.set_tick_params(labelbottom=False)
axis[0,1].yaxis.set_tick_params(labelleft=False)
plt.show()
