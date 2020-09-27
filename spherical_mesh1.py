'''
This code takes the density values in spherical grid at r,theta,phi and plots the polar mesh for density distribution
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import matplotlib.colors as colors
import pickle
from multiprocessing import Pool
pool = Pool()


r_coord=np.loadtxt('radius.txt',dtype=float)
t_coord=np.loadtxt('theta.txt',dtype=float)
p_coord=np.loadtxt('phi.txt',dtype=float)
print (max(r_coord),min(r_coord))

den_sph_file = open('density_sph_100.pkl', 'rb') 
density_sph = pickle.load(den_sph_file)
den_sph_file.close()	

den_sph=density_sph[:,:,120]  
dr= 0.006  #kpc
dt= 0.0175 #rad
print (den_sph[612][90])
den_sph=den_sph  #here row is varied so data is storedin cloumn and then to next...but in mesh x means column is varied so take transpose
t,r=np.meshgrid(p_coord,r_coord)     # y and x
#print (t.min(),t.max())
#print(r[0][1],t[0][1])

l_den_sph=np.log10(den_sph)  #log of density

fig=plt.figure()

cmap = plt.get_cmap('jet')

ax=plt.subplot(111,projection='polar')
#ax.set_xlim(t.min(), t.max())
ax.set_ylim(r.min(),r.max())
#plt.xlabel('X[kpc]',fontsize=12)
#plt.ylabel('Y[kpc]',fontsize=12)
#ax.set_xticks(np.array([t.min(),t.max()/2,t.max()]))
c=plt.pcolormesh(t,r,l_den_sph, cmap=cmap, vmin=np.min(l_den_sph),vmax=np.max(l_den_sph))
cbar=plt.colorbar(c)
cbar.ax.set_ylabel(r'$\log(\mathrm{n[cm^{-3}]})$',size=14)
#cbar.tick_params(labelsize=10) 
#plt.title(r'Density plot (R-$\phi(\theta\approx 90)$)', fontweight ="bold",fontsize=14)
plt.tight_layout()
#plt.grid()
fig.savefig('r-t_p_0_100.png',dpi=300)
plt.show()
		
