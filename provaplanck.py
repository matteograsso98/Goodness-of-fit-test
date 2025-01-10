import numpy as np 
# read some spectra to pass to the likelihood (can use CAMB/CLASS to generate these)
home = '/Users/matteograsso/Desktop/'
ls, Dltt, Dlte, Dlee, Dlbb = np.genfromtxt(home + 'hi_class_public/output_planck/cl.dat', unpack=True)
ellmin=int(ls[0])
#print('ls', ls[0:10])
#print('TT', Dltt[0:10])

#convert model Dl's to Cls then bin them
ls=np.arange(len(Dltt))+ellmin
fac=ls*(ls+1)/(2*np.pi)
Cltt=Dltt/fac

home = '/Users/matteograsso/Desktop/planck_p_value/data'
binw_file = np.loadtxt(home + '/planck2018_plik_lite/bweight.dat')

bin_w_low_ell=np.loadtxt(home + '/planck2018_low_ell/bweight_low_ell.dat')
blmin_low_ell=np.loadtxt(home +'/planck2018_low_ell/blmin_low_ell.dat').astype(int)
blmin=np.loadtxt(home + '/planck2018_plik_lite/blmin.dat').astype(int)

blmin_TT=np.concatenate((blmin_low_ell, blmin+len(bin_w_low_ell))) 


#print(len(binw_file)) 
#print(len(bin_w_low_ell))

#print(blmin_TT) #numbers from 0 to 7432; looks like np.logspace
print(blmin_TT[0]+ 30 - 2)

#Cltt_bin[i]=np.sum(Cltt[self.blmin_TT[i]+ 30 - 2
                        #:self.blmax_TT[i]+self.plmin_TT+1-ellmin]*self.bin_w_TT[self.blmin_TT[i]:self.blmax_TT[i]+1])