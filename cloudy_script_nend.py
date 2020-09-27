
'''
This python script is to run cloudy for different hydrogen density profiles along the radius. The data output from python txt file is log of radius den.
A script file is already written and will be modified for different densities and correspondingly input files will be created and cloudy result will be saved in the folder cloudy_spherical_in. Cloudy is run over all the files and .ovr and .out are saved in the same folder.
python cloudy_script_nend.py init_file file-name ncores cores-number files start end
'''


import os
import sys
import numpy as np
from multiprocessing import Pool
from functools import partial
import multiprocessing


#***********************************************************************

def input_files(start,files):
	files_dir ='/mnt/home/student/cmeenakshi/public/spherical_den-new_236'
	save_path='/mnt/home/student/cmeenakshi/public/cloudy_den-new_236' 
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

def cloudy(input_f):
	os.chdir('/mnt/home/student/cmeenakshi/public/cloudy_den-new_236')
	os.system('cloudy'+' '+input_f)

if __name__ == '__main__':
	files_dir ='/mnt/home/student/cmeenakshi/public/spherical_den-new_236'  
	save_path='/mnt/home/student/cmeenakshi/public/cloudy_den-new_236'   #directory where input files for cloudy will be saved
	argList=sys.argv
	files_list=[]

	if 'init_file' in argList:
		initIdx=argList.index('init_file')
		init_file=argList[initIdx+1]
		file_name=init_file+'.in'
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
		with open(file_name,'r') as file:  #reading init file to find index to append the input file for density variation and inner radius
			for line_no, line in enumerate(file):
				if 'luminosity' in line:
					r_indx= line_no+1   #to add inner radius of the cloud
				if 'dlaw table' in line:
					start_indx=line_no+2   #density table start index
					break
		
	with open('input_236.txt','r') as my_file:
		f_list=[]
		for lines in my_file:	
			i=lines.split('.txt')
			f_n=str(i[0])
			i1=f_n.split('_')
			#print (f_n,i1)
			#if i1[2]=='0' or i1[4]=='90':
			f_list.append(str(i[0])+'.txt')

	files_list=f_list[a:b]		
	
	func=partial(input_files,start_indx)
	pool = Pool(processes = num_cores)


	input_list=pool.map(func,files_list)

	pool.map(cloudy,input_list)
	pool.close()     


