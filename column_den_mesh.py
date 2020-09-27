'''

This file creates the mesh of cumulative column density

'''

import numpy as np
import os
#from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import matplotlib.colors as colors
import pickle
from scipy.integrate import simps
from functools import partial
from multiprocessing import Pool
import multiprocessing

phi=np.loadtxt('phi.txt',dtype=float)
theta=np.loadtxt('theta.txt',dtype=float)
radius=np.loadtxt('radius.txt',dtype=float)
p=len(phi)
t=len(theta)
r=len(radius)
kpc=3.086e21
rad=radius[0]*kpc  #radius in cm
'''

#file_dir1='/mnt/home/student/cmeenakshi/public/cloudy_spherical_100_l_45'
file_dir2='/mnt/home/student/cmeenakshi/public/spherical_den_236'   #spherical_den   and spherical_den_236
col_den=np.zeros((r,t,p)) #array for storing column desnity at each point of mesh

files_list=[]
for file_n in os.listdir(file_dir2):  #file for .txt of density profile
	files_list.append(file_n)


#print (len(files_list))
print ('hi')
def col_den_find(file_n2):
	cd={}
	#file_dir2='/mnt/home/student/cmeenakshi/spherical/spherical_den'
	with open(os.path.join(file_dir2,file_n2)) as f:
		#print (file_n2)
		radius_l,h_l=np.loadtxt(f,usecols=(0,1),unpack=True)
		radius_l=10**radius_l
		h_l=10**h_l
		a=file_n2.split(r"_")
		ph=int(a[2])
		th=int(a[4].split(".")[0])
		cd['LOS_'+str(th)+'_'+str(ph)]=[]
		for i in range(1,r):
			h=h_l[:i+1]
			d=radius_l[:i+1]
			c_d=simps(h,d)  #integration till each grid point for LOS
			col_den[i][th][ph]=c_d
			cd['LOS_'+str(th)+'_'+str(ph)].append([i,col_den[i][th][ph]])
		return cd




num_cores=multiprocessing.cpu_count()
#func = partial(col_den_find,dep)
#den=Parallel(n_jobs=num_cores,verbose=10)(delayed(func)(idx) for idx in indx)
pool = Pool(processes = num_cores)
val=pool.map(col_den_find,files_list)
#print (len(val))
pool.close()

print ('h')
for l in range(len(val)):
	for name,el in val[l].items():
		name_d=name.split(r'_')
		th=int(name_d[1])
		ph=int(name_d[2])
		col_den[0][th][ph]=0.1
		for i in range(len(el)):	
			col_den[i+1][th][ph]=el[i][1]


del val


den_sph_file= open('col_den_236.pkl', 'wb') 

pickle.dump(col_den, den_sph_file)
den_sph_file.close()
'''
dbfile = open('col_den_100.pkl','rb') #density values in cartesian coordinates
density_ion = pickle.load(dbfile)
dbfile.close()
print (density_ion.shape)
den_sph=density_ion[:,90,:] 
l_den_sph=np.log10(den_sph)
print(density_ion[0][1][2],density_ion[1][1][2],density_ion[4][1][2])
t,r=np.meshgrid(theta,radius)     #angle(x) and radius(y)
#print (t)
#print (r)

fig=plt.figure()

cmap = plt.get_cmap('rainbow')
print (np.min(den_sph))
ax=plt.subplot(111,projection='polar')
#ax.set_xlim(t.min(), t.max())
ax.set_ylim(r.min(),3)
c=plt.pcolormesh(t,r,l_den_sph, cmap=cmap,vmin=18,vmax=np.max(l_den_sph))
cbar=plt.colorbar(c)
cbar.ax.set_ylabel(r'$\log(\mathrm{n[cm^{-2}]})$',fontsize=12)
#plt.title(r'Column density(R-$\theta(\phi\approx 0)$)', fontweight ="bold")
plt.tight_layout()
#plt.grid()
fig.savefig('r-p_t_90_cd_100.png',dpi=300)

plt.show()


















