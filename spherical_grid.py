

'''
This code creates the radius theta and phi grid points and save them to txt files.
Then it values from r.theta,file txt file and save it to a list of dictionaries which is then dumped to a binary file and will be used by spherical_interp.py.
'''

import numpy as np
import pickle,json
import time

#*************************Spherical grid is created**************************

z=np.loadtxt('gridz.out',usecols=(0),unpack=True)
y=np.loadtxt('gridy.out',usecols=(0),unpack=True)
x=np.loadtxt('gridx.out',usecols=(0),unpack=True)


r_1=np.sqrt(np.min(abs(z))**2+np.min(abs(x))**2+np.min(abs(y))**2)
r_2=np.sqrt(np.max(abs(z))**2+np.max(abs(x))**2+np.max(abs(y))**2)

print (r_1,r_2)
radius=[]
radius.append(r_1)
while(r_1<r_2):
	if (r_1<2.5):
		r_1=r_1+0.006
	else:
		r_1=r_1+0.012
	radius.append(r_1)

#r_mid=np.median(radius)
#print (r_mid)
#dt=(0.006/r_mid)*2.0
#print (dt)
dt=0.0175  # 1 degrees 
theta=[]
t=0
while(t<=2*np.pi):
	theta.append(t)
	t=t+dt
	
print (t)		

phi=[]
p=0 

while(p<=2*np.pi):
	phi.append(p)
	p=p+dt  #0.172 degrees max resolution at r_2 is 0.06 pcs
	

radius=np.array(radius)
theta=np.array(theta)
#theta=theta[:-1]
phi=np.array(phi)
print (len(radius),len(theta),len(phi))
np.savetxt('radius.txt',radius,fmt='%0.4f')
np.savetxt('theta.txt',theta,fmt='%0.4f')
np.savetxt('phi.txt',phi,fmt='%0.4f')


#***********************Spherical grid coordinates loaded in a list of dictionary and then dumped to a binary file******************

start_time=time.time()
np.loadtxt('radius.txt',dtype='float')
np.loadtxt('theta.txt',dtype='float')
np.loadtxt('phi.txt',dtype='float')
r=len(radius)
t=len(theta)
p=len(phi)
print (r,t,p)
cord={}
l=0
#data=np.zeros(r*t*p)
for k in range(p):   # p ==180 is fine for the whole grid, 360 overlaps many data points, but takeing all 360 points mkes it convenient to plot the mesh at different planes, say \phi =0 to 180, so it covers the whole grid for interpolation
#take p= 360, but theta=180 and plot the two halfs separately
	for j in range(t):
		for i in range(r):
			cord['cell_'+str(l)]=[radius[i],theta[j],phi[k]]
			l+=1

			
data=[cord]

spherical_cord=open('spherical_cordinate.pkl','wb')
pickle.dump(data,spherical_cord)
spherical_cord.close()
print (time.time()-start_time)











