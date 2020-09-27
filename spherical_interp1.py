'''
This code saves the density values at spherical grid from trilinear interpolation of cartesin values in a binary file.(Parallel processing used, n=4)
'''

import numpy as np
import pickle
from multiprocessing import Pool
from functools import partial
#from joblib import Parallel, delayed
import time
import multiprocessing

start_time = time.time()
mid=0.072
radius=np.loadtxt('radius.txt',usecols=(0),unpack=True)
theta=np.loadtxt('theta.txt',usecols=(0),unpack=True)
phi=np.loadtxt('phi.txt',usecols=(0),unpack=True)

r=len(radius)
t=len(theta)
p=len(phi)
density_sph=np.zeros((r,t,p))
num_cores=multiprocessing.cpu_count()
print (density_sph.shape)

z_coord=np.loadtxt('gridz.out',usecols=(0),unpack=True)
y_coord=np.loadtxt('gridy.out',usecols=(0),unpack=True)
x_coord=np.loadtxt('gridx.out',usecols=(0),unpack=True)
a=len(x_coord)-1
b=len(y_coord)-1
c=len(z_coord)-1

dbfile = open('density_100.pkl','rb') #density values in cartesian coordinates
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
	if (z<z_c[0]):C=mid#xd=yd=zd=0;x0=x1=x_l;y0=y1=y_l;z0=z1=z_l  #points below torus(towards negative z-axis)  densitty =0.04750

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
#den=[]
for i in range(len(sph_grid[0])):
	indx.append(sph_grid[0]['cell_'+str(i)])
#	den.append(density_sph_grid(x_coord,y_coord,z_coord,a,b,c,density,indx[i]))	
	
#indx=[[0.0052030, 0.0, 0.0],[0.01230279, 0.0, 0.0]]
#print (len(den))
del sph_grid
pool = Pool(processes = num_cores)
func = partial(density_sph_grid, x_coord,y_coord,z_coord,a,b,c)
#den=Parallel(n_jobs=num_cores,verbose=10)(delayed(func)(idx) for idx in indx)
den=pool.map(func,indx)
pool.close()	
num=0

del den_data
den=np.array(den)
for k in range(p):
	for j in range(t):
		for i in range(r):
			density_sph[i][j][k]=den[num]	
			num=num+1	 	

den_sph_file = open('density_sph_100.pkl', 'wb') 

pickle.dump(density_sph, den_sph_file)
den_sph_file.close()
				
print(time.time() - start_time)				
				
			
			
			
