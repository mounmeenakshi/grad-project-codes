'''
This code saves temperature profile in pickle file..plots the mesh and so the interpolation for spherical grid.

'''
import numpy as np
import matplotlib.pyplot as plt
import pickle
from multiprocessing import Pool
from functools import partial
#from joblib import Parallel, delayed
import time
import multiprocessing
import statistics
import os
import math

f=np.fromfile('temp_236.flt','<f4')  #density file #little endian
#f.sort()
#mid=np.median(f)
#print (mid)
print (np.min(f),np.max(f))

kpc=3.086e21 #kpc to cm
z=np.loadtxt('gridz.out',usecols=(0),dtype=float)
y=np.loadtxt('gridy.out',usecols=(0),dtype=float)
x=np.loadtxt('gridx.out',usecols=(0),dtype=float)
'''
print (np.min(f),np.max(f))
db={}
print (len(z),len(y),len(x))
dbfile = open('temp_236.pkl', 'wb') 
density=np.zeros((len(x),len(y),len(z)))
l=0
for k in range(len(z)):
	for j in range(len(y)):
		for i in range(len(x)):
			density[i][j][k]=f[l]
			#db['']=density[i][j][k]
			l=l+1
			

pickle.dump(density, dbfile)
dbfile.close()


#****************Temperature mesh plot *******************************

print (np.min(x),np.max(x))
dbfile = open('temp_236.pkl','rb')
density = pickle.load(dbfile)
dbfile.close()
print(density.shape)	

z_m=density[:,:,392]
#zm_m, x_m = np.mgrid[slice(z[0], z[(len(z)-1)], dz), slice(x[0], x[(len(x)-1)], dx)] 
z_m=z_m.T       #here row is varied so data is storedin cloumn and then to next...but in mesh x means column is varied so take transpose
x_m,y_m=np.meshgrid(x,y)     
#print (x_m)
#print(zm_m)
#print(x_m[0][1],y_m[0][1])
l_z_m=np.log10(z_m)  #log of density
print (np.min(l_z_m),np.max(l_z_m))
#print(x_m,y_m)
#print (z_m)
fig=plt.figure()
#plt.xlim(x_m.min(), x_m.max())
plt.xlabel('X[kpc]')
plt.ylabel('Y[kpc]')
#plt.ylim(y_m.min(), y_m.max())
##plt.axis('scaled')
cmap = plt.get_cmap('plasma')
#norm = colors.Normalize(vmin=np.min(l_z_m),vmax=np.max(l_z_m))
c=plt.pcolormesh(x_m, y_m, l_z_m, cmap=cmap, vmin=np.min(l_z_m),vmax=np.max(l_z_m))
plt.colorbar(c,label=r'$\log(\mathrm{T[K]})$')
#plt.title('Temp plot (X-Y) Plane', fontweight ="bold")
plt.tight_layout()
fig.savefig('X-Y_temp_236.pdf',dpi=200)
plt.show()


# ***************Interpolation*********************8
start_time = time.time()
print ('h')
radius=np.loadtxt('radius.txt',usecols=(0),unpack=True)
theta=np.loadtxt('theta.txt',usecols=(0),unpack=True)
phi=np.loadtxt('phi.txt',usecols=(0),unpack=True)

r=len(radius)
t=len(theta)
p=len(phi)
density_sph=np.zeros((r,t,p))
num_cores=multiprocessing.cpu_count()

z_coord=np.loadtxt('gridz.out',usecols=(0),unpack=True)
y_coord=np.loadtxt('gridy.out',usecols=(0),unpack=True)
x_coord=np.loadtxt('gridx.out',usecols=(0),unpack=True)
a=len(x_coord)-1
b=len(y_coord)-1
c=len(z_coord)-1

dbfile = open('temp_236.pkl','rb') #density values in cartesian coordinates
den_data = pickle.load(dbfile)
dbfile.close()

sph_cord=open('spherical_cordinate.pkl','rb') #spherical coordinates list of dictionaries
sph_grid=pickle.load(sph_cord)
sph_cord.close()

#print (sph_grid[0]['cell_'+str(0)])

tot=(len(sph_grid[0]))
#print (tot)

def sph_to_cart(r1,t1,p1):    #function to convert spherical to cartesian coordinates
	x11=r1*np.sin(t1)*np.cos(p1)
	y11=r1*np.sin(t1)*np.sin(p1)
	z11=r1*np.cos(t1)
	return x11,y11,z11 

  #***********************Finding indices for the cube where the x,y,z point lies*******************
a=len(x_coord)-1
b=len(y_coord)-1
c=len(z_coord)-1

def find_x_indices(x11,x_co):
	for l in range(a):
		if (x_co[l]<x11) and (x_co[l+1]>x11):
			break
	return l,l+1

def find_y_indices(y11,y_co):
	for l in range(b):
		if (y_co[l]<y11) and (y_co[l+1]>y11):
			break
	return l,l+1

def find_z_indices(z11,z_co):
	for l in range(c):
		if (z_co[l]<z11) and (z_co[l+1]>z11):
			break
	return l,l+1
 
 #***************Filling the spherical density array using trilinear interpolation*********************8888

#result=[]
def density_sph_grid(x_c,y_c,z_c,x_l,y_l,z_l,list_cord):
	rad=list_cord[0]
	th=list_cord[1]
	ph=list_cord[2]
	x,y,z=sph_to_cart(rad,th,ph)
	#print (x,x_coord[0],y,y_coord[0],z,z_coord[0])
	if (z<z_c[0]):C=mid #points below torus(towards negative z-axis)  densitty =0.04750

	elif (z>z_c[c]):C=mid #points above torus
	
	elif(y_c[b]<y):C=mid  #y-coordinate is beyond range
	elif(y_c[0]>y):C=mid
		
	elif(x_c[a]<x):C=mid
	elif(x_c[0]>x):C=mid
				
			
	else:                  # all lies in the given range of mesh
		x0,x1=find_x_indices(x,x_c)
		y0,y1=find_y_indices(y,y_c)
		z0,z1=find_z_indices(z,z_c)
		xd=(x-x_c[x0])/(x_c[x1]-x_c[x0])
		yd=(y-y_c[y0])/(y_c[y1]-y_c[y0])
		zd=(z-z_c[z0])/(z_c[z1]-z_c[z0])
				
		C00=den_data[x0][y0][z0]*(1-xd)+den_data[x1][y0][z0]*xd
		C01=den_data[x0][y0][z1]*(1-xd)+den_data[x1][y0][z1]*xd
		C10=den_data[x0][y1][z0]*(1-xd)+den_data[x1][y1][z0]*xd
		C11=den_data[x0][y1][z1]*(1-xd)+den_data[x1][y1][z1]*xd
		C0=C00*(1-yd)+C10*yd
		C1=C01*(1-yd)+C11*yd
		C=C0*(1-zd)+C1*zd
		#result.append(C) 
	return C
	

indx=[]

for i in range(len(sph_grid[0])):
	indx.append(sph_grid[0]['cell_'+str(i)])
#	den.append(density_sph_grid(x_coord,y_coord,z_coord,a,b,c,density,indx[i]))	

del sph_grid
print('h')
pool = Pool(processes = num_cores)
func = partial(density_sph_grid, x_coord,y_coord,z_coord,a,b,c)
#den=Parallel(n_jobs=num_cores,verbose=10)(delayed(func)(idx) for idx in indx)
den=pool.map(func,indx)
pool.close()	
num=0

del den_data
den=np.array(den)

print('h')
for k in range(p):
	for j in range(t):
		for i in range(r):
			density_sph[i][j][k]=den[num]	
			num=num+1	 	

den_sph_file = open('temp_sph_100.pkl', 'wb') 

pickle.dump(density_sph, den_sph_file)
den_sph_file.close()
				
print(time.time() - start_time)	


#*************Saving radial profiles for temperature*******************
radius=np.loadtxt('radius.txt',dtype=float)
theta=np.loadtxt('theta.txt',dtype=float)
phi=np.loadtxt('phi.txt',dtype=float)

r=len(radius)
t=len(theta)
p=len(phi)

tmp_sph_file = open('temp_sph_100.pkl', 'rb') 
tmp_sph = pickle.load(tmp_sph_file)
tmp_sph_file.close()	

tmp_sph=tmp_sph[:,90,:]  
dr= 0.006  #kpc
dt= 0.0175 #rad


t,r=np.meshgrid(phi,radius)     # y and x
#print (t.min(),t.max())
#print(r[0][1],t[0][1])

l_tmp_sph=np.log10(tmp_sph)  #log of density

fig=plt.figure()

cmap = plt.get_cmap('plasma')

ax=plt.subplot(111,projection='polar')
#ax.set_xlim(t.min(), t.max())
ax.set_ylim(r.min(), 3)
#ax.set_xticks(np.array([t.min(),t.max()/2,t.max()]))
c=plt.pcolormesh(t,r,l_tmp_sph, cmap=cmap, vmin=np.min(l_tmp_sph),vmax=np.max(l_tmp_sph))
plt.colorbar(c,label=r'$\log(\mathrm{T[K]})$')
#plt.title(r'Density plot (R-$\phi(\theta\approx 90)$)', fontweight ="bold")
plt.tight_layout()
#plt.grid()
fig.savefig('r-p_t_90_tmp-new_100.jpg',dpi=200)
plt.show()
	

#****This part saves the temperature profile along radius in a txt file in a folder************

print ('hi')
dbfile = open('temp_sph_100.pkl','rb') #density values in cartesian coordinates
tmp_profile = pickle.load(dbfile)
print (tmp_profile.shape)
dbfile.close()
radius=np.loadtxt('radius.txt',dtype=float)
theta=np.loadtxt('theta.txt',dtype=float)
phi=np.loadtxt('phi.txt',dtype=float)

r=len(radius)
t=len(theta)
p=len(phi)


os.chdir('/mnt/home/student/cmeenakshi/public/spherical_temp_100_45_new')
radius=radius*kpc   #radius in cm (for cloudy input file)
for k in range(p):
	for j in range(t):
		if (j==int(90)):
			temp=np.zeros(r)
			for i in range(r):
				temp[i]=tmp_profile[i][j][k]
			
			data=np.column_stack((np.log10(radius),np.log10(temp)))  #radius(distance from centre source) in cm and den in /cm^{-3}
			np.savetxt('den_p_' +str(k)+'_t_'+str(j)+'.txt',data)
		#print ('hi')
print ('hi')	

	

#**********Changing density due to temperature profile ****************************		


file_dir1='/mnt/home/student/cmeenakshi/public/spherical_temp_236_45'	#folder with temp file
file_dir2='/mnt/home/student/cmeenakshi/public/spherical_den_236' #folder with earlier den file
file_dir3='/mnt/home/student/cmeenakshi/public/spherical_den-new_236'#folder to save new files

file_list=[]

for file_n in os.listdir(file_dir1):
	file_list.append(file_n)


h=6.626*10**(-27) #Planck constant
m=9.1*10**(-28) #mass of electron
k=1.38*10**(-16) # Boltzmann constant
lmbda=(h**2/(2*np.pi*m*k))**(3/2)

delta_E=13.6*1.6*10**(-12)/(1.38*10**(-16)) #  (E/K)

def den_files(files):
	with open(os.path.join(file_dir1,files)) as fp:
		rad1,tmp= np.loadtxt(fp,usecols=(0,1),unpack=True)
		filename=files
		hden_filename = os.path.join(file_dir3, filename) 
	with open(os.path.join(file_dir2,filename)) as f:
		rad2,hden= np.loadtxt(f,usecols=(0,1),unpack=True)
	hden_new=np.zeros(len(hden))
	for i in range(len(rad2)):
		value=((10**tmp[i])**(3/2))*math.exp(-delta_E/10**tmp[i])/(lmbda*10**hden[i])
		if (value>=10000.0):  # 50 percent ionization when value is 0.5 = (x^2/1-x)
			hden_new[i]=-4.177
		else:	
			ion_frac=(-value+np.sqrt(value**2+4*value))/2
			hden_1=(10**hden[i])*(1-ion_frac)
			hden_new[i]=np.log10(hden_1)

	data=np.column_stack((rad2,hden_new))
	np.savetxt(hden_filename,data)

num_cores=32
pool = Pool(processes = num_cores)
pool.map(den_files,file_list)	
pool.close()
print ('hi')


file_dir3='/mnt/home/student/cmeenakshi/public/spherical_den-new_236' #folder for new den file
radius=np.loadtxt('radius.txt',dtype=float)
theta=np.loadtxt('theta.txt',dtype=float)
phi=np.loadtxt('phi.txt',dtype=float)

r=len(radius)
t=len(theta)
p=len(phi)
density=np.zeros((r,t,p))
for file_n in os.listdir(file_dir3):
	a=file_n.split('.txt')
	b=a[0].split('_')
	ph=int(b[2])
	th=int(b[4])
	with open(os.path.join(file_dir3,file_n)) as f:
		rad,hden= np.loadtxt(f,usecols=(0,1),unpack=True)
	for i in range(len(rad)):
		density[i][th][ph]=hden[i]  #saving log files 

print ('hi')
dbfile = open('den_new_236_new.pkl', 'wb') 
pickle.dump(density, dbfile)
dbfile.close()



#*************Saving new modified radial profiles for density*************88

dbfile = open('den_new_236_new.pkl','rb') 
den_profile = pickle.load(dbfile)
print (den_profile.shape)
dbfile.close()

radius=np.loadtxt('radius.txt',dtype=float)
theta=np.loadtxt('theta.txt',dtype=float)
phi=np.loadtxt('phi.txt',dtype=float)

r=len(radius)
t=len(theta)
p=len(phi)	

den_profile=den_profile[:,90,:]  
dr= 0.006  #kpc
dt= 0.0175 #rad


t,r=np.meshgrid(phi,radius)     # y and x
#print (t.min(),t.max())
#print(r[0][1],t[0][1])

#l_tmp_sph=np.log10(den_profile)  #log of density

fig=plt.figure()

cmap = plt.get_cmap('jet')

ax=plt.subplot(111,projection='polar')
#ax.set_xlim(t.min(), t.max())
ax.set_ylim(r.min(), 3)
#ax.set_xticks(np.array([t.min(),t.max()/2,t.max()]))
c=plt.pcolormesh(t,r,den_profile, cmap=cmap,vmin=-4.177)# vmin=np.min(den_profile),vmax=np.max(den_profile)
cbar=plt.colorbar(c)
cbar.ax.set_ylabel(r'$\log(\mathrm{n[cm^{-3}]})$',size=14)

#plt.title(r'Density plot (R-$\phi(\theta\approx 90)$)', fontweight ="bold")
plt.tight_layout()
#plt.grid()
fig.savefig('r-p_t_90_den-new_236.png',dpi=300)
plt.show()
'''
