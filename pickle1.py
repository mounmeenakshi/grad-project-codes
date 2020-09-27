import numpy as np
import pickle

f=np.fromfile('rho_236.flt','<f4')  #density file #little endian
#np.savetxt('binary.txt',f)
f.sort()
print (np.median(f))
print (np.mean(f))
#print (len(f))
#print (np.min(f),np.max(f))
db={}
'''
dbfile = open('density_236.pkl', 'wb') 
density=np.zeros((672,672,784))
l=0
for k in range(784):
	for j in range(672):
		for i in range(672):
			density[i][j][k]=f[l]
			#db['']=density[i][j][k]
			l=l+1
			

pickle.dump(density, dbfile)
dbfile.close()


dbfile = open('density_236.pkl','rb')
density1 = pickle.load(dbfile)

dbfile.close()
print(density.shape)
np.array_equal(density,density1)
'''	
	
