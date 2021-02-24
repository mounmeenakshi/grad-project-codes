
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
start_time=time.time()


x=np.loadtxt('gridx.out',usecols=(0),unpack=True)
y=np.loadtxt('gridy.out',usecols=(0),unpack=True)
z=np.loadtxt('gridz.out',usecols=(0),unpack=True)

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


if 'num' in argList:
	file_Idx=argList.index('num')
	file_s=argList[file_Idx+1]
	file_e=argList[file_Idx+2]
	a=int(file_s)
	b=int(file_e)
#ph=0.1 #in degrees angle with x-axis (theta between zero and 360),
#print (ph)
th=45
ph=89.9

phi=float(ph)*np.pi/180  #in radians


#th=0.1 #in degrees angle with z-axis (theta between zero and 180),
theta=float(th)*np.pi/180  #in radians
#print (theta)

n_x=np.sin(theta)*np.cos(phi)
n_y=np.sin(theta)*np.sin(phi)
n_z=np.cos(theta)


dbfile = open('/mnt/home/student/cmeenakshi/public/OIII_236.pkl','rb') #density values in cartesian coordinates
den_data = pickle.load(dbfile)
dbfile.close()
den_data=den_data.T
#print (den_data.shape)


###############################################################################################################33


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_last=x[len(x)-1]
y_last=y[len(y)-1]
z_last=z[len(z)-1]

#img_plane=np.zeros((len(x),len(y)))


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




def intrsct_pt(i1,i2,i3):  #function to find the intersection point with the grid planes
	y_0=y[0]
	d=abs((y_0-i2)/n_y)
	y0=y_0
	yn=y0+(n_y*delta)
	x0=i1-(n_x*d)
	z0=i3-(n_z*d)
	#print (x0,yn,z0)
	if (x_last<x0) or (x0<x[0]) or (z[0]>z0) or (z0>z_last) or (y[0]>yn) or (yn>y_last):
		#print ('h')
		y_0=y_last
		d=abs((y_0-i2)/n_y)
		y0=y_0
		yn=y0+(n_y*delta)
		x0=i1-(n_x*d)
		z0=i3-(n_z*d)
		#print (x0,z0,yn)
		if (x_last<x0) or (x0<x[0]) or (z[0]>z0) or (z0>z_last) or (y[0]>yn) or (yn>y_last):
			x_0=x[0]
			d=abs((x_0-i1)/n_x)
			#print ('d')
			x0=x_0
			xn=x0+(n_x*delta)
			y0=i2-(n_y*d)
			z0=i3-(n_z*d)
			if (y[0]>y0) or (y0>y_last) or (z[0]>z0) or (z0>z_last) or (x[0]>xn) or (xn>x_last):
				x_0=x_last
				d=abs((x_0-i1)/n_x)
				#print (d)
				x0=x_0
				xn=x0+(n_x*delta)
				y0=i2-(n_y*d)
				z0=i3-(n_z*d)
				if (y[0]>y0) or (y0>y_last) or (z[0]>z0) or (z0>z_last) or (x[0]>xn) or (xn>x_last):
					z_0=z[0]
					#print ('h')
					d=abs((z_0-i3)/n_z)
					#print (d)
					z0=z_0
					zn=z0+(n_z*delta)
					y0=i2-(n_y*d)
					x0=i1-(n_x*d)
					#print (x0,y0,zn)
					if (y[0]>y0) or (y0>y_last) or (x[0]>x0) or (x0>x_last) or (z[0]>zn) or (zn>z_last):
						z_0=z_last
						#print ('h')
						d=abs((z_0-i3)/n_z)
						#print (d)
						z0=z_0
						zn=z0+(n_z*delta)
						y0=i2-(n_y*d)
						x0=i1-(n_x*d)
						if (y[0]>y0) or (y0>y_last) or (x[0]>x0) or (x0>x_last) or (z[0]>zn) or (zn>z_last):
							x0=y0=z0='no intersect'
							
	
	return x0,y0,z0					



def interp_grid(x_c,y_c,z_c,list_cord):  #function to interpolate the values
	values=[]
	for ele in range(len(list_cord)):
		x_val=list_cord[ele][0]
		y_val=list_cord[ele][1]
		z_val=list_cord[ele][2]
		if (len(list_cord)==1):
			#print (list_cord)
			values=[0,0,0]
			#lum=1e-50
		else:

			if (x_val<=x[0]):
				x0=x1=0;
				xd=1;
			elif (x_val>=x_last):
				x0=x1=len(x)-1;
				xd=1;
			else:
				indx=np.argmin(abs(x_val-x))
				#print (indx)
				if (x_val>x[indx]):
					x0= indx;x1=indx+1;
				else:
					x0=indx-1;x1=indx;
		
				xd=(x_val-x[x0])/(x[x1]-x[x0])
			if (y_val<=y[0]):
				y0=y1=0;
				yd=1;
			elif (y_val>=y_last):
				y0=y1=len(y)-1;
				yd=1;
			else:
				indx=np.argmin(abs(y_val-y))
				#print (indx)
				if (y_val>y[indx]):
					y0= indx;y1=indx+1;
				else:
					y0=indx-1;y1=indx;
				yd=(y_val-y[y0])/(y[y1]-y[y0])

			if (z_val<=z[0]):
				z0=z1=0;
				zd=1;
			elif (z_val>=z_last):
				z0=z1=len(z)-1;
				zd=1;
			else:
				indx=np.argmin(abs(z_val-z))
				#print (indx)
				if (z_val>z[indx]):
					z0= indx;z1=indx+1;
				else:
					z0=indx-1;z1=indx;
				zd=(z_val-z[z0])/(z[z1]-z[z0])
			#print (xd,yd,zd)
			C00=den_data[x0][y0][z0]*(1-xd)+den_data[x1][y0][z0]*xd
			C01=den_data[x0][y0][z1]*(1-xd)+den_data[x1][y0][z1]*xd
			C10=den_data[x0][y1][z0]*(1-xd)+den_data[x1][y1][z0]*xd
			C11=den_data[x0][y1][z1]*(1-xd)+den_data[x1][y1][z1]*xd
			C0=C00*(1-yd)+C10*yd
			C1=C01*(1-yd)+C11*yd
			C=C0*(1-zd)+C1*zd 
			values.append(C)
			#break
	#if (len(values)<2):
	#print (values)
	lum=delta*sum(values)-(delta*(values[0]+values[len(values)-1])*0.5)  #trapz. integration
	return lum
	

delta=0.006
R=np.sqrt(x_last**2+y_last**2+z_last**2) #this is the perpendicular distance to the initial position of image plane, also the radius of the circle on which this image plane rotates, this takes all the gris points inside the integrating zone

plane=[None]*len(z)  #rows is the no of line
t=0
for row in range(len(z)): #z axis is the no of rows
	line=[None]*(len(y))
	for col in range(len(y)):  #column is determined by the y-axis
		xa1,ya1,za1=rotate(z[len(z)-1-row],y[col],R)  
		points=[xa1,ya1,za1]
		line[col]=points
		

	plane[t]=line
	t=t+1

#print ('h')
img=[]
plane_short=plane[a:b]
del plane

for lines in range(len(plane_short)):  #no of lines
	#lum_line=[None]*len(y)  #no of points in a line
	#i=0
	plane_f=[]
	for pt in plane_short[lines]:  #  the points in the line
		i_1=pt[0];i_2=pt[1];i_3=pt[2];
		#print (i_1,i_2,i_3)
		xl,yl,zl=intrsct_pt(i_1,i_2,i_3) #to find the intersection point
		#print (type(xl))
		if (type(xl)==str):
			line_f=[[xl,yl,zl]]
			
		else:
			pt_0=[xl,yl,zl]
			#print (pt0)
			#break
			line_f=[]
			line_f.append(pt_0)
			a=True
			m=0
			while (a==True):
				m=m+1
				xf=m*n_x*delta+xl
				yf=m*n_y*delta+yl
				zf=m*n_z*delta+zl
				
				if (x[0]<=xf<=x_last) and (z[0]<=zf<=z_last) and (y[0]<=yf<=y_last):
					pts=[xf,yf,zf]
					line_f.append(pts) #the points lying on the line which is the LOS for the point on line
				else: 
					a=False

	#Interpolation for the new points in the plane
		plane_f.append(line_f) #this contains the poins lying along the LOS , which needs to be interpolated
	num_cores=32
	pool = Pool(processes = num_cores)
	func = partial(interp_grid, x,y,z)
	den=pool.map(func,plane_f)   #integrated values at the image plane
	pool.close()	
	img.append(den)
	#print (img[0][0])


pkl_file = open('img_%d.pkl' % b, 'wb')  #saving pickle files for plane set

pickle.dump(img, pkl_file)
pkl_file.close()

time_take=time.time() - start_time

f=open('log.out','w')
f.write("Job is done with time {}\n".format(time_take))
f.close()




'''
dbfile = open('cyl_imag_90.pkl', 'rb') #density values in cartesian coordinates
img_plane = pickle.load(dbfile)
dbfile.close()
print (img_plane.shape)
z_m=img_plane[:,:]
print (img_plane[4,:])
#zm_m, x_m = np.mgrid[slice(z[0], z[(len(z)-1)], dz), slice(x[0], x[(len(x)-1)], dx)] 
l_z_m=z_m      #here row is varied so data is storedin cloumn and then to next...but in mesh x means column is varied so take transpose
pl=np.linspace(1,672,672)
print (len(pl))
x_m,y_m=np.meshgrid(y,x)     
#print (x_m)
#print(zm_m)
#print(x_m[0][1],y_m[0][1])
#l_z_m=np.log10(z_m)  #log of density
print (np.min(l_z_m),np.max(l_z_m))
#print(x_m,y_m)
#print (z_m)
fig=plt.figure()
plt.xlim(x_m.min(), x_m.max())
plt.xlabel('X[kpc]',fontsize=12)
plt.ylabel('Y[kpc]',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax = fig.add_subplot(111)
ax.set_aspect('equal')
#plt.ylim(y_m.min(), y_m.max())
#plt.axis('scaled')
cmap = plt.get_cmap('jet')
#norm = colors.Normalize(vmin=np.min(l_z_m),vmax=np.max(l_z_m))
c=plt.pcolormesh(x_m, y_m, l_z_m, cmap=cmap, vmin=np.min(l_z_m),vmax=np.max(l_z_m))
cbar=plt.colorbar(c)
cbar.ax.set_ylabel(r'$\log(\mathrm{n[cm^{-3}]})$',fontsize=14)
plt.title('density plot (X-Y) Plane', fontweight ="bold",fontsize=14)
plt.tight_layout()
fig.savefig('X-Y_100_c.png',dpi=300)
plt.show()


print(time.time() - start_time)


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

q=0
for line in plane:
	#q=q+1
	a=[]
	b=[]
	c=[]
	for p in range(len(line)):
		a.append(line[p][0])
		b.append(line[p][1])
		c.append(line[p][2])
			
	ax.plot3D(a,b,c,linewidth=2)#zdir='z',s=20)
	#plt.show()

plt.show()
#plt.grid()		

#print (plane[700][600])

'''


