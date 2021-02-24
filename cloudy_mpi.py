
'''
This python script is to run cloudy for different hydrogen density profiles along the radius. The data output from python txt file is log of radius den.
A script file is already written and will be modified for different densities and correspondingly input files will be created and cloudy result will be saved in the folder cloudy_spherical_in. Cloudy is run over all the files and .ovr and .out are saved in the same folder.
python cloudy_script.py init_file hden_int ncores 32 files start end
'''


import os
import sys
import numpy as np
from mpi4py import MPI
Comm = MPI.COMM_WORLD
rank=Comm.Get_rank()
size=Comm.Get_size()
#***************************************************************************
'''
files_dir ='/mnt/home/student/cmeenakshi/public/folders/spherical_den_236_shock_new'

files_list=[]
for file_n in os.listdir(files_dir):
	files_list.append(str(file_n))

print (len(files_list))
with open('input_cloudy.txt', 'w') as f:
	for item in files_list:
		f.write("%s\n" % item)
'''
#***********************************************************************

def split(container, count):
	return [container[_i::count] for _i in range(count)]


'''
def input_files(Comm,start,files):
	#for f1 in range(len(fil)):
	files_dir ='/mnt/home/student/cmeenakshi/public/folders/spherical_den_236'
	save_path='/mnt/home/student/cmeenakshi/public/folders/cloudy_den_236_mpi' 
	with open(os.path.join(files_dir,files)) as fp:
		radius,hden= np.loadtxt(fp,usecols=(0,1),unpack=True)
		base=os.path.basename(files)
		os.path.splitext(base)	
		filename=os.path.splitext(base)[0] #to put name to the input file for cloudy	
		#print (filename)
		hden_filename = os.path.join(save_path, filename)   
		a=start    
	with open(file_name,'r') as file, open(hden_filename+'.in','w') as file1:
		filedata=file.readlines()
		filedata.insert(r_indx,'radius'+' '+ str(radius[0])+'\n')	
		for i in range(len(hden)):
			filedata.insert(a,str(radius[i])+' '+str(hden[i])+'\n')  #adding density table 
			a+=1
		file1.write("".join(filedata))	
	return filename			
	a=start	


#********************************************************************************************************************


def cloudy(Comm,input_f):
	os.chdir('/mnt/home/student/cmeenakshi/public/folders/cloudy_den_236_mpi')
	os.system('cloudy'+' '+input_f)

'''
if __name__ == '__main__':
	files_dir ='/mnt/home/student/cmeenakshi/public/folders/spherical_den_100_shock_new'  
	save_path='/mnt/home/student/cmeenakshi/public/folders/syf_100'   #directory where input files for cloudy will be saved
	argList=sys.argv
	files_list=[]

	if rank==0:
		f=open('log.out','w')
		f.write("Job has been initiated from process {} out of {} processes\n".format(rank,size))
		f.close()
	if 'init_file' in argList:
		initIdx=argList.index('init_file')
		init_file=argList[initIdx+1]
		file_name=init_file+'.in'
	#if 'ncores' in argList:
	#	ncores_Idx=argList.index('ncores')
	#	num=argList[ncores_Idx+1]
	#	num_cores=int(num)
	if 'files' in argList:
		file_Idx=argList.index('files')
		file_s=argList[file_Idx+1]
		file_e=argList[file_Idx+2]
		a=int(file_s)
		b=int(file_e)
		#print (file_name)
		with open(file_name,'r') as file:  #reading init file to find index to append the input file for density variation and inner radius
			for line_no, line in enumerate(file):
				#if 'luminosity' in line:
					#r_indx= line_no+1   #to add inner radius of the cloud
				if 'dlaw table' in line:
					start_indx=line_no+1   #density table start index
					break
		
	with open('input_cloudy.txt','r') as my_file:
		f_list=[]
		for lines in my_file:	
			i=lines.split('.txt')
			f_n=str(i[0])
			i1=f_n.split('_')
			if i1[4]=='262':
			#print (f_n,i1)
			#if i1[2]=='0' or i1[4]=='90':
				f_list.append(str(i[0])+'.txt')
		

	#if COMM.rank==0:
	#files_list=f_list[a:b]
	#print (COMM.size)
	#if COMM.rank == 0:
	jobs=split(f_list,Comm.size)
	jobs = Comm.scatter(jobs, root=0)
	#print (jobs)
	#func=partial(input_files,start_indx)
	#input_list=[]
	#print (jobs)input_list

	input_list=[]
	for job in jobs:
		#print (job)
		#for k in job:
		#for f1 in range(len(fil)):
		os.chdir('/mnt/home/student/cmeenakshi/public')
		files_dir ='/mnt/home/student/cmeenakshi/public/folders/spherical_den_100_shock_new'
		save_path='/mnt/home/student/cmeenakshi/public/folders/syf_100' 
		with open(os.path.join(files_dir,job)) as fp:
			radius,hden= np.loadtxt(fp,usecols=(0,1),unpack=True)
			base=os.path.basename(job)
			os.path.splitext(base)	
			filename=os.path.splitext(base)[0] #to put name to the input file for cloudy	
			#print (filename)
			hden_filename = os.path.join(save_path, filename)   
			a=start_indx   
		with open(file_name,'r') as file, open(hden_filename+'.in','w') as file1:
			filedata=file.readlines()
			#filedata.insert(r_indx,'radius'+' '+ str(radius[0])+'\n')	
			for i in range(len(hden)):
				filedata.insert(a,str(radius[i])+' '+str(hden[i])+'\n')  #adding density table 
				a+=1
			file1.write("".join(filedata))	
		os.chdir('/mnt/home/student/cmeenakshi/public/folders/syf_100')
		os.system('cloudy'+' '+filename)	
		#cloudy(Comm,input_list)
	   

