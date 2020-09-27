

'''
This code saves the cartesian coordinate in a pkl file. These coordinates will be used to interpolate the values from spherical to cartesian grid.
'''

import numpy as np
import pickle,json
from multiprocessing import Pool
from functools import partial
#from joblib import Parallel, delayed
import multiprocessing
#*************************Spherical grid is created**************************
'''
z=np.loadtxt('gridz.out',usecols=(0),unpack=True)
y=np.loadtxt('gridy.out',usecols=(0),unpack=True)
x=np.loadtxt('gridx.out',usecols=(0),unpack=True)
p=len(z)
t=len(y)
r=len(x)

cord={}
l=0
#data=np.zeros(r*t*p)
for k in range(p):
	for j in range(t):
		for i in range(r):
			cord['cell_'+str(l)]=[x[i],y[j],z[k]]
			l+=1

			
data=[cord]

cart_cord=open('cartesian_cordinate.pkl','wb')
pickle.dump(data,cart_cord)
cart_cord.close()
print (time.time()-start_time)
'''
#************************Interpolation starts here *************************************

radius=np.loadtxt('radius.txt',usecols=(0),unpack=True)
theta=np.loadtxt('theta.txt',usecols=(0),unpack=True)
phi=np.loadtxt('phi.txt',usecols=(0),unpack=True)

z_coord=np.loadtxt('gridz.out',usecols=(0),unpack=True)
y_coord=np.loadtxt('gridy.out',usecols=(0),unpack=True)
x_coord=np.loadtxt('gridx.out',usecols=(0),unpack=True)

a=len(x_coord)
b=len(y_coord)
c=len(z_coord)

r=len(radius)
t=len(theta)
p=len(phi)
lum_cart=np.zeros((a,b,c))
num_cores=multiprocessing.cpu_count()
print (lum_cart.shape)



dbfile = open('density_sph_100.pkl','rb') #luminosity values in spherical grid
lum_data = pickle.load(dbfile)
dbfile.close()

cart_cord=open('cartesian_cordinate.pkl','rb') #cartesian coordinates list of dictionaries
cart_grid=pickle.load(cart_cord)
cart_cord.close()

#print (cart_grid[0]['cell_'+str(0)])

#tot=(len(cart_grid[0]))
#print (tot)
import math


def cart_to_sph(x1,y1,z1):    #function to convert spherical to cartesian coordinates
	r11=np.sqrt(x1**2+y1**2+z1**2)
	t11=math.acos(z1/r11)
	p11=(math.atan(y1/x1))
	#print (r11,t11,p11)
	return r11,t11,p11 

  #***********************Finding indices for the sphere the r,theta, phi point lies in the region of sphere*******************

def find_r_indices(r11,r_co):
	for l in range(r):
		if (r_co[l]<r11) and (r_co[l+1]>r11):
			break
	return l,l+1

def find_t_indices(t11,t_co):
	for l in range(t):
		if (t_co[l]<t11) and (t_co[l+1]>t11):
			break
	return l,l+1

def find_p_indices(p11,p_co):
	for l in range(p):
		if (p_co[l]<p11) and (p_co[l+1]>p11):
			break
	return l,l+1
 
 #***************Filling the spherical density array using trilinear interpolation*********************8888

#result=[]
def cart_lum_grid(r_c,t_c,p_c,list_cord):
	x_c=list_cord[0]
	y_c=list_cord[1]
	z_c=list_cord[2]
	rad,th,ph=cart_to_sph(x_c,y_c,z_c)
	if (rad>radius[r-1]):
		C=0.047	 
	
	else:
		r0,r1=find_r_indices(rad,r_c)
		t0,t1=find_t_indices(th,t_c)
		p0,p1=find_p_indices(ph,p_c)
		rd=(rad-r_c[r0])/(r_c[r1]-r_c[r0])
		td=(th-t_c[t0])/(t_c[t1]-t_c[t0])
		pd=(ph-p_c[p0])/(p_c[p1]-p_c[p0])
				
		C00=lum_data[r0][t0][p0]*(1-rd)+lum_data[r1][t0][p0]*rd
		C01=lum_data[r0][t0][p1]*(1-rd)+lum_data[r1][t0][p1]*rd
		C10=lum_data[r0][t1][p0]*(1-rd)+lum_data[r1][t1][p0]*rd
		C11=lum_data[r0][t1][p1]*(1-rd)+lum_data[r1][t1][p1]*rd
		C0=C00*(1-td)+C10*td
		C1=C01*(1-td)+C11*td
		C=C0*(1-pd)+C1*pd
	return C
	

indx=[]

l=0
#data=np.zeros(r*t*p)
#for k in range(c):
#	for j in range(b):
#		for i in range(a):
#			indx.append([x_coord[i],y_coord[j],z_coord[k]])
#			l+=1

			

#den=[]
for i in range(len(cart_grid[0])):
	indx.append(cart_grid[0]['cell_'+str(i)])
	#den.append(density_sph_grid(x_coord,y_coord,z_coord,a,b,c,density,indx[i]))	
	
#indx=[[0.0052030, 0.0, 0.0],[0.01230279, 0.0, 0.0]]
#print (len(den))
pool = Pool(processes = 32)
func = partial(cart_lum_grid,radius,theta,phi)
#den=Parallel(n_jobs=num_cores,verbose=10)(delayed(func)(idx) for idx in indx)
lum=pool.map(func,indx)
pool.close()	
num=0

del lum_data
lum=np.array(lum)
for k in range(c):
	for j in range(b):
		for i in range(a):
			lum_cart[i][j][k]=lum[num]	
			num=num+1	 	

lum_cart_file = open('cart_lum_100.pkl', 'wb') 

pickle.dump(lum_cart,lum_cart_file)
lum_cart_file.close()
				
			
				
			
			
			









