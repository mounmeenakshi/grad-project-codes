A
'''
This code calculates the luminosity integrated on the image plane. The image plane is rotated by \theta and \phi and then the LOS are mapped in the grid to integrate and find the values on the image plane.

'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from multiprocessing import Pool
from functools import partial
#from joblib import Parallel, delayed
import time
#import multiprocessing
import os
import pickle
import sys
import scipy.interpolate as spint
import pymp
import pymp.shared


RGI = spint.RegularGridInterpolator
start_time=time.time()

x,x_len=np.loadtxt('/mnt/home/student/cmeenakshi/public/gridx.out',usecols=(0,1),unpack=True)
y,y_len=np.loadtxt('/mnt/home/student/cmeenakshi/public/gridy.out',usecols=(0,1),unpack=True)
z,z_len=np.loadtxt('/mnt/home/student/cmeenakshi/public/gridz.out',usecols=(0,1),unpack=True)


'''
ph=0.1 #in degrees angle with x-axis (theta between zero and 360),
phi=ph*np.pi/180  #in radians
print (phi)

th=0.1 #in degrees angle with z-axis (theta between zero and 180),
theta=th*np.pi/180  #in radians
print (theta)

'''

argList=sys.argv
files_list=[]

if 'th' in argList:
	initIdx=argList.index('th')
	th=argList[initIdx+1]
	th=float(th)
	

if 'phi' in argList:
	initIdx=argList.index('phi')
	ph=argList[initIdx+1]
	ph=float(ph)


if 'name' in argList:
	ncores_Idx=argList.index('name')
	num=argList[ncores_Idx+1]
	name=int(num)


if 'psi' in argList:
	ncores_Idx=argList.index('psi')
	num_1=argList[ncores_Idx+1]
	ps=float(num_1)



phi=float(ph)*np.pi/180  #in radians


#th=0.1 #in degrees angle with z-axis (theta between zero and 180),
theta=float(th)*np.pi/180  #in radians
#print (theta)
print (phi,theta)
n_x=np.sin(theta)*np.cos(phi)
n_y=np.sin(theta)*np.sin(phi)
n_z=np.cos(theta)


dbfile = open('/mnt/home/student/cmeenakshi/P45_dir00_new/P45_dir00/folders/45_deg/S-Xray_%d.pkl' % name,'rb') #density values in cartesian coordinates
#dbfile = open('/mnt/home/student/cmeenakshi/settle_nw200/O_%d.pkl' % name,'rb')
emiss = pickle.load(dbfile)
dbfile.close()

###############################################################################################################33


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_last=x[len(x)-1]
y_last=y[len(y)-1]
z_last=z[len(z)-1]

#img_plane=np.zeros((len(x),len(y)))


psi=float(ps)*np.pi/180 


def rotate(x_11,y_11,z_11):
	#************Rotating by theta about y-axis************
	x_1=np.cos(theta)*x_11+np.sin(theta)*z_11  
	y_1=y_11
	z_1=-np.sin(theta)*x_11+np.cos(theta)*z_11

	#************Rotating by phi about z-axis************

	x_2=np.cos(phi)*x_1-np.sin(phi)*y_1
	y_2=np.sin(phi)*x_1+np.cos(phi)*y_1
	z_2=z_1


	return (x_2,y_2,z_2)


def rodrigues(x_v,y_v,z_v):
	x_mi=0.0;y_mi=0.0;z_mi=0.0;
	x_r=((x_v-x_mi)*np.cos(psi))+((n_y*(z_v-z_mi)-n_z*(y_v-y_mi))*np.sin(psi))+((1-np.cos(psi))*((x_v-x_mi)*n_x+(y_v-y_mi)*n_y+(z_v-z_mi)*n_z)*n_x)
	y_r=((y_v-y_mi)*np.cos(psi))+((n_z*(x_v-x_mi)-n_x*(z_v-z_mi))*np.sin(psi))+((1-np.cos(psi))*((x_v-x_mi)*n_x+(y_v-y_mi)*n_y+(z_v-z_mi)*n_z)*n_y)
	z_r=((z_v-z_mi)*np.cos(psi))+((n_x*(y_v-y_mi)-n_y*(x_v-x_mi))*np.sin(psi))+((1-np.cos(psi))*((x_v-x_mi)*n_x+(y_v-y_mi)*n_y+(z_v-z_mi)*n_z)*n_z)

	return (x_r,y_r,z_r)



def intrsct_pt(i1,i2,i3):  #function to find the intersection point with the grid planes
	y_0=y[0]
	d=-(y_0-i2)/n_y
	y0=y_0
	yn=y0+(n_y*delta)
	x0=i1-(n_x*d)
	z0=i3-(n_z*d)
	#print (x0,yn,z0)
	if (x_last<x0) or (x0<x[0]) or (z[0]>z0) or (z0>z_last) or (y[0]>yn) or (yn>y_last):
		#print ('h')
		y_0=y_last
		d=-(y_0-i2)/n_y
		y0=y_0
		yn=y0+(n_y*delta)
		x0=i1-(n_x*d)
		z0=i3-(n_z*d)
		#print (x0,z0,yn)
		if (x_last<x0) or (x0<x[0]) or (z[0]>z0) or (z0>z_last) or (y[0]>yn) or (yn>y_last):
			x_0=x[0]
			d=-(x_0-i1)/n_x
			#print ('d')
			x0=x_0
			xn=x0+(n_x*delta)
			y0=i2-(n_y*d)
			z0=i3-(n_z*d)
			if (y[0]>y0) or (y0>y_last) or (z[0]>z0) or (z0>z_last) or (x[0]>xn) or (xn>x_last):
				x_0=x_last
				d=-(x_0-i1)/n_x
				#print (d)
				x0=x_0
				xn=x0+(n_x*delta)
				y0=i2-(n_y*d)
				z0=i3-(n_z*d)
				if (y[0]>y0) or (y0>y_last) or (z[0]>z0) or (z0>z_last) or (x[0]>xn) or (xn>x_last):
					z_0=z[0]
					#print ('h')
					d=-(z_0-i3)/n_z
					#print (d)
					z0=z_0
					zn=z0+(n_z*delta)
					y0=i2-(n_y*d)
					x0=i1-(n_x*d)
					#print (x0,y0,zn)
					if (y[0]>y0) or (y0>y_last) or (x[0]>x0) or (x0>x_last) or (z[0]>zn) or (zn>z_last):
						z_0=z_last
						#print ('h')
						d=-(z_0-i3)/n_z
						#print (d)
						z0=z_0
						zn=z0+(n_z*delta)
						y0=i2-(n_y*d)
						x0=i1-(n_x*d)
						if (y[0]>y0) or (y0>y_last) or (x[0]>x0) or (x0>x_last) or (z[0]>zn) or (zn>z_last):
							x0=y0=z0='no intersect'
							
	
	return x0,y0,z0					




rgi = RGI(points=[x,y,z],values=emiss)  # function for interpolation



def interp_grid(list_cord):  #function to interpolate the values
	values=rgi(list_cord)
	#print (values)
	lum=delta*sum(values)-(delta*(values[0]+values[len(values)-1])*0.5)  #trapz. integration
	#print (lum)
	return lum

	

delta=0.006
R=np.sqrt(x_last**2+y_last**2+z_last**2) #this is the perpendicular distance to the initial position of image plane, also the radius of the circle on which this image plane rotates, this takes all the gris points inside the integrating zone

x_value1=[None]*len(z)*len(z)  #rows is the no of line
y_value1=[None]*len(z)*len(z)
z_value1=[None]*len(z)*len(z)
row_indx=[None]*len(z)*len(z)
col_indx=[None]*len(z)*len(z)
t=0
for row in range(len(z)): #z axis is the no of rows, in other direction (x posiitve to negative)
	#line=[None]*(len(z))
	for col in range(len(z)):  #column is determined by the y-axis
		xa1,ya1,za1=rotate(z[len(z)-1-row],z[col],R)  # from fixed z  , x and z fixed and y is changing
		x_value1[t]=xa1
		y_value1[t]=ya1
		z_value1[t]=za1
		row_indx[t]=row
		col_indx[t]=col
		t=t+1



x_value= [None]*len(z)*len(z)  
y_value=[None]*len(z)*len(z)
z_value=[None]*len(z)*len(z)

for i in range(len(x_value)):
	xa1,ya1,za1=rodrigues(x_value1[i],y_value1[i],z_value1[i])
	x_value[i]=xa1
	y_value[i]=ya1
	z_value[i]=za1


del x_value1
del y_value1
del z_value1	


'''
ax.view_init(5,60)
ax.set_xlim3d(-2,2)
ax.set_ylim3d(-2,2)
ax.set_zlim3d(-4,4)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xticks((-2,-1,0,1,2))
ax.set_yticks((-2,-1,0,1,2))
ax.set_zticks((-4,-3,-2,-1,0,1,2,3,4))
x_value=x_value[0:24960]
y_value=y_value[0:24960]
z_value=z_value[0:24960]
ax.plot3D(x_value,y_value,z_value,linewidth=2)#zdir='z',s=20)
#plt.show()

plt.show()
#plt.grid()		

#print (plane[700][600])
'''
density_array = pymp.shared.array((len(z),len(z)))
#density_array = pymp.shared.array((len(z),len(z)))

x_value1=np.reshape(x_value,(784,784))
y_value1=np.reshape(y_value,(784,784))
z_value1=np.reshape(z_value,(784,784))
print (x_value1[783,783])
print (y_value1[783,783])
print (z_value1[783,783])
with pymp.Parallel(32) as q:
	#for k in q.range(len(z_value)):  # z_value is total number of cells in the image plane
		  #values in the image plane
		#density_array[row_indx[k],col_indx[k]]=0.0
		for i in q.range(len(z)):
			for j in range(len(z)):
				xl,yl,zl=intrsct_pt(x_value1[i,j],y_value1[i,j],z_value1[i,j])
				#if (i==783) and (type(xl)!=str):	print (xl,yl,zl,x_value1[i,j],y_value1[i,j],z_value1[i,j],d1)
				if (type(xl)!=str):
					#line_f=[[xl,yl,zl]]
					pt_0=[xl,yl,zl]
					#print (pt0)
					#break
					line_f=[]
					line_f.append(pt_0)
					m=1
					xf=m*n_x*delta+xl
					yf=m*n_y*delta+yl
					zf=m*n_z*delta+zl
					while (x[0]<xf<x_last) and (z[0]<zf<z_last) and (y[0]<yf<y_last):
						pts=[xf,yf,zf]
						line_f.append(pts)
						m=m+1
						xf=m*n_x*delta+xl
						yf=m*n_y*delta+yl
						zf=m*n_z*delta+zl
			
					points=np.array(line_f)
				
					#density_array[row_indx[k],col_indx[k]]=interp_grid(points)#*z_len[row_indx[k]]*z_len[col_indx[k]] 					
					if len(points)>0:
						density_array[i,j]=interp_grid(points)
			#print (density_array[row_indx[k],col_indx[k]])

pkl_file = open('sxray_t%s_p%s_ps%s_%d.pkl' % (str(int(th)),str(int(ph)),str(int(ps)),name), 'wb')  #saving pickle files for plane set

pickle.dump(density_array, pkl_file)
pkl_file.close()

time_take=time.time() - start_time

f=open('log.out','w')
f.write("Job is done with time {}\n".format(time_take))
f.close()
