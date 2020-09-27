
'''
This code reads the .ovr files for each LOS(only two planes we want to plot here) and then creates a 3-D array which stores boolean values(1 and 0) for ionized and non-ionized regions.
And plots polar mesh for ionized region.
'''

import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
from multiprocessing import Pool
import multiprocessing

kpc=3.086e21
phi=np.loadtxt('phi.txt',dtype=float)
theta=np.loadtxt('theta.txt',dtype=float)
radius=np.loadtxt('radius.txt',dtype=float)
p=len(phi)
t=len(theta)
r=len(radius)

num_cores=32


rad=radius[0]  #inner radius of the cloud
rad=rad*kpc #inner cloud radius in cms

file_again=[]
start_time = time.time()

files_name=[]
file_dir ='/mnt/home/student/cmeenakshi/public/cloudy_den-new_100'

for file_n2 in os.listdir(file_dir):
		if file_n2.endswith('.ovr'):
			files_name.append(file_n2)
print (len(files_name))

with open('input_ovr.txt', 'w') as f:
	for item in files_name:
		f.write("%s\n" % item)

print (files_name[0])
'''
#*****************C

def add_again(file_n):
	files_dir ='/mnt/home/student/cmeenakshi/public/cloudy_den-new_100'
	reason=[]
	with open(os.path.join(files_dir,file_n)) as file: 
		for line_no, line in enumerate(file):
			if 'requested radius outside range of dense_tabden' in line:
				reason.append(file_n)
				break
					#file_again.append(a[0])
					#print (a[0])
			#elif line_no>1000:
			#	reason.append(file_n)
			#	break
	return reason
				


#print (files_name[0],files_name[1])
pool = Pool(processes = num_cores)
val=pool.map(add_again,files_name)
pool.close()

with open('file_again_236.txt', 'w') as f:
	for item in val:
		if (item!=[]):
			f.write("%s\n" % item[0])



#**************************If cloudy stops due to something else reason then above*******8

dist=np.zeros((t,p)) 

def distance(file_n):
	cd={}
	#print (file_n)
	with open(os.path.join(file_dir,file_n)) as f:
		a=file_n.split(r"_")
		ph=int(a[2])
		th=int(a[4].split(".")[0])
		#print (file_n)
		cd['LOS_'+str(th)+'_'+str(ph)]=[]	
		depth,hden=np.loadtxt(f,usecols=(0,3),unpack=True)
		dist[th][ph]=depth[len(depth)-1]+rad 
		dist[th][ph]=depth[len(depth)-1]+rad  #the farthest distance for ionization(from the central source))
		cd['LOS_'+str(th)+'_'+str(ph)].append([dist[th][ph]])
		
	return cd

with open('file_again_236.txt', 'r') as f:
	f_list=[]
	for lines in f:	
		i=lines.split('.out')
		f_list.append(str(i[0]))


with open('input_ovr.txt','r') as my_file:
	f1_list=[]
	for lines in my_file:	
		i=lines.split('.ovr')
		f1_list.append(str(i[0]))

print (len(f1_list),len(f_list))
files_name=[]

val1=[]
for i in f1_list:
	if i not in f_list:
		files_name.append(str(i)+'.ovr')
	else:
		name=i.split('_')
		th=int(name[4])
		ph=int(name[2])
		cd={}
		cd['LOS_'+str(th)+'_'+str(ph)]=[]
		cd['LOS_'+str(th)+'_'+str(ph)].append([radius[r-1]*kpc])
		val1.append(cd)

print (len(val1))
print(val1[0])	
pool = Pool(processes = num_cores)
val2=pool.map(distance,files_name)
pool.close()
#print (val)
#print (val[0],val[1],val[2])
val3=val1+val2
print (val2[0])
spherical_cord=open('last_100-new.pkl','wb')
pickle.dump(val3,spherical_cord)
spherical_cord.close()


print ('hi')
spherical_cord=open('last_100-new.pkl','rb')
val=pickle.load(spherical_cord)
spherical_cord.close()
del spherical_cord
print (len(val))
print (val[0],val[1])
density_ion=np.zeros((r,t,p))
for l in range(len(val)):
	for name,el in val[l].items():
		name_d=name.split(r'_')
		th=int(name_d[1])
		ph=int(name_d[2])
		#print (th,ph,el[0][0])
		#break
		for i in range(r):
			if (kpc*radius[i]<=el[0][0]):
				density_ion[i][th][ph]=1   #the region is ionized to 50% or more
			else:	
				density_ion[i][th][ph]=0 
			


				

den_sph_file= open('density_ion_100-new.pkl', 'wb') 

pickle.dump(density_ion, den_sph_file)
den_sph_file.close()

print(time.time() - start_time)
'''
spherical_cord=open('last_100-new.pkl','rb')
val=pickle.load(spherical_cord)
spherical_cord.close()

Area=0
num=0
for l in range(len(val)):
	for name,el in val[l].items():
		name_d=name.split(r'_')
		th=int(name_d[1])
		ph=int(name_d[2])
		if (th==90) and (el[0][0]<radius[r-1]*kpc):
			num=num+1    #how many LOS in a plane
			Area=Area+(el[0][0])**2   #Total areas
		  #EFfective radius in the disk plane
print (num)
eff_rad=np.sqrt(Area/num)/kpc
print (eff_rad)
eff_rad=np.ones(len(phi))*eff_rad

dbfile = open('density_ion_100-new.pkl','rb') #density values in cartesian coordinates
density_ion = pickle.load(dbfile)
dbfile.close()
print (density_ion.shape)
den_sph=density_ion[:,90,:] 
print(density_ion[4][190][0],density_ion[45][190][0])
t,r=np.meshgrid(phi,radius)     #angle(x) and radius(y)
#print (t)
#print (r)

fig=plt.figure()

cmap = plt.get_cmap('YlGnBu')
print (np.min(den_sph))
ax=plt.subplot(111,projection='polar')
#ax.set_xlim(t.min(), t.max())
ax.set_ylim(r.min(),3)
c=plt.pcolormesh(t,r,den_sph, cmap=cmap,vmin=np.min(den_sph),vmax=np.max(den_sph))
ax.plot(phi,eff_rad,color='r',label='$%.3f\,\mathrm{kpc}$' %eff_rad[0])
#plt.colorbar(c)
#plt.title(r'Ionized region plot (R-$\theta(\phi\approx 0)$)', fontweight ="bold")
plt.tight_layout()
plt.grid(linestyle='dotted')
ax.legend(bbox_to_anchor=(1,1.05))
fig.savefig('ion_t_90_100-new.png',dpi=300)
plt.show()

