
'''
THis code finds the caretsain coordinate for sphericla and divide them in categories based on z-axis and then integrate in the categories for luminosity.

'''

import numpy as np
import pickle,json
from multiprocessing import Pool
from functools import partial
#from joblib import Parallel, delayed
import multiprocessing
#from scipy.integrate import simps as sim
import math
import sys
import matplotlib.pyplot as plt


'''
#*************THis part saves the mesh for the plot, we have taken average for the y and z values we defined above*********************
z_array1=[]
y_array1=[]

for i in range(len(z_r)-1):
	z_array1.append((z_r[i]+z_r[i+1])*0.5)

z_array1=np.array(z_array1)

for i in range(len(y_r)-1):
	y_array1.append((y_r[i]+y_r[i+1])*0.5)

y_array1=np.array(y_array1)
#print (len(z_r),len(z_array1))
np.savetxt('gridz_new.out',z_array1,fmt='%.4f')
np.savetxt('gridy_new.out',y_array1,fmt='%.4f')
#print (z_r[0],z_r[1],z_r[len(z_r)-1])

#***********************************************************************88
'''


kpc=3.086e21  #kpc to cm


def sph_to_cart(r1,t1,p1):    #function to convert spherical to cartesian coordinates
	x11=r1*np.sin(t1)*np.cos(p1)
	y11=r1*np.sin(t1)*np.sin(p1)
	z11=r1*np.cos(t1)
	return x11,y11,z11 


def find_z_range(z_cod,z_rr):
	for m1 in range(len(z_rr)-1):
		if (z_cod>=z_rr[m1]) and (z_cod<z_rr[m1+1]):
			break
	return m1


def find_y_range(y_cod,y_rr):
	for n1 in range(len(y_rr)-1):
		if (y_cod>=y_rr[n1]) and (y_cod<y_rr[n1+1]):
			break
	return n1

def start(list_cord):  #start the computation by this fucntion in parallel computing
	rad=list_cord[0]
	th=list_cord[1]
	ph=list_cord[2]
	lumsty=list_cord[3]
	x1,y1,z1=sph_to_cart(rad,th,ph)
	n=find_y_range(y1,y_r)
	m=find_z_range(z1,z_r)
	list_add=[n,m,x1,lumsty]
	return list_add


def area(dict_1):  #function to find the points lying in particular bin and adding them to integarte for the luminosity
	#print ('h')
	d_name=dict_1.split('_')
	n=d_name[1]
	m=d_name[2]
	new_list=sorted(zip(globals()["H_"+str(n)+'_'+str(m)]['x'],globals()["H_"+str(n)+'_'+str(m)]['lum']))
	lum_array=[]
	dist_array=[]
	for lnth in new_list:
		dist_array.append(lnth[0])
		lum_array.append(lnth[1])
		#print (lnth[0])
	#print (new_list)
	if (len(lum_array)>1):
		lum_val=np.trapz(lum_array,dist_array)*kpc
	elif (len(lum_array)==1):
		if (dist_array[0]>0.0):
			dist_array.insert(0,-float(dist_array[0]))
			lum_array.insert(0,lum_array[0])

		if (dist_array[0]<0.0):
			dist_array.append(abs(float(dist_array[0])))
			lum_array.append(lum_array[0])

		if (dist_array[0]==0.0):
			dist_array.insert(0,-2.017)
			lum_array.insert(0,lum_array[0])
			dist_array.append(2.017)
			lum_array.append(lum_array[0])
				
		lum_val=(np.trapz(lum_array,dist_array))*kpc

	else:
		lum_val= 10**(-50)*kpc*4.034
	
	#print (lum_val)
	
	return lum_val
				 	
if __name__ == '__main__':
	argList=sys.argv

	if 'ncores' in argList:
		ncores_Idx=argList.index('ncores')
		num=argList[ncores_Idx+1]
		num_cores=int(num)
	if 'files' in argList:
		file_Idx=argList.index('files')
		file_s=argList[file_Idx+1]
		file_e=argList[file_Idx+2]
		a=int(file_s)
		b=int(file_e)

	radius=np.loadtxt('radius.txt',usecols=(0),unpack=True)
	theta=np.loadtxt('theta.txt',usecols=(0),unpack=True)
	phi=np.loadtxt('phi.txt',usecols=(0),unpack=True)

	r=len(radius)
	t=len(theta)
	p=len(phi)

	dbfile = open('H-alpha_100.pkl','rb') #luminosity values in spherical grid
	lum_data1= pickle.load(dbfile)
	dbfile.close()
	#print (lum_data1.shape)
	l=0

	z=np.loadtxt('gridz.out',usecols=(0),unpack=True)
	y=np.loadtxt('gridy.out',usecols=(0),unpack=True)
	x=np.loadtxt('gridx.out',usecols=(0),unpack=True)
	z_len=len(z)
	y_len=len(y)
	x_len=len(x)


	z_r=np.arange(z[0],z[z_len-1],0.012)
	y_r=np.arange(y[0],y[y_len-1],0.012)
	'''
	cord=[]
	for k in range(1,p):
		for j in range(1,181):
			for i in range(r):
				list1=[radius[i],theta[j],phi[k],lum_data1[i][j][k]]
				cord.append(list1)
	

	for i in range(r):
		list1=[radius[i],theta[0],phi[0],lum_data1[i][j][k]]
		cord.append(list1)
	
	pool = Pool(processes = num_cores)
	den = pool.map(start,cord)
	#print (den)
	pool.close()
	
	spherical_cord=open('H-alpha_100_list.pkl','wb')
	pickle.dump(den,spherical_cord)
	spherical_cord.close()	
	

	dict_list=[]

	for i in range(len(z_r)-1):
		for j in range(len(y_r)-1):
			#globals()['H_'+str(j)+'_'+str(i)]={}  
			dict_list.append(str('H_'+str(j)+'_'+str(i)))	

	with open('dict_lum.txt', 'w') as f:  #writing dictionaries to a file
		for item in dict_list:
			f.write("%s\n" % item)
	'''
	sph_cord=open('H-alpha_100_list.pkl','rb') #spherical coordinates list of dictionaries
	sph_grid=pickle.load(sph_cord)
	sph_cord.close()
	tot=(len(sph_grid))

	for i in range(len(z_r)-1):
		for j in range(len(y_r)-1):
			globals()['H_'+str(j)+'_'+str(i)]={}  
			globals()["H_"+str(j)+'_'+str(i)]['lum']=[]
			globals()["H_"+str(j)+'_'+str(i)]['x']=[]	

	for alist in sph_grid:
		n11=alist[0]
		m11=alist[1]
		globals()["H_"+str(n11)+'_'+str(m11)]['x'].append(alist[2])
		globals()["H_"+str(n11)+'_'+str(m11)]['lum'].append(alist[3])
		

	
	#print (sph_grid[0],sph_grid[30])
	
	
	with open('dict_lum.txt','r') as my_file:
		dict1_list=[]
		for lines in my_file:	
			i=lines.split('\n')
			dict1_list.append(str(i[0]))


	dict_list=dict1_list[a:b]
	#print ('h')
	#func = partial(area,sph_grid)	
	pool = Pool(processes = num_cores)
	lum_list = pool.map(area,dict_list)
	pool.close()
	lum_list=np.array(lum_list)
	np.savetxt(str(a)+'.txt',np.c_[lum_list])
	#print (lum_list)
	#lum_data=np.ones((len(y_r)-1,len(z_r)-1))
	#num=0


	#for i in range(len(z_r)-1):
	#	for j in range(len(y_r)-1):
	#		lum_data[j][i]=lum_list[num]
	#		num=num+1
	#lum_list1=np.column_stack((lum_list))
	
	#print (len(lum_list))
	#print (num)
	#print (lum_list)

	#lum_file = open('H-alpha_100_proj2.pkl', 'wb') 

	#pickle.dump(lum_data,lum_file)
	#lum_file.close()
	

'''
dbfile = open('H-alpha_100_proj.pkl','rb') #ion , 0 for non-ionized, 1 for ionized
lumnsity= pickle.load(dbfile)
dbfile.close()
print (np.min(lumnsity),np.max(lumnsity))
y=np.loadtxt('gridy_new.out')
z=np.loadtxt('gridz_new.out')
t,r=np.meshgrid(z,y)     # y and x

l_tmp_sph=np.log10(lumnsity)  #log of density

fig=plt.figure()

cmap = plt.get_cmap('viridis')

#ax=plt.subplot(111,projection='polar')
#ax.set_ylim(r.min(), 3)
#ax.set_xticks(np.array([t.min(),t.max()/2,t.max()]))
c=plt.pcolormesh(t,r,l_tmp_sph,cmap=cmap)#,vmin=l_lim,vmax=h_lim
plt.title(r'$\mathrm{H\alpha}$ Emission')
plt.colorbar(c,label=r'$\log \,\mathrm{L[erg\, cm^{-3}\, s^{-1}]}$')
#plt.title(r'Density plot (R-$\phi(\theta\approx 90)$)', fontweight ="bold")
plt.tight_layout()
#plt.grid()
fig.savefig('H-alpha_100_proj.png',dpi=200)
plt.show()

'''








