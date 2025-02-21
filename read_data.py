import os, sys
import numpy as np  
from math import radians, degrees

sin = np.sin  
cos = np.cos 
exp = np.exp 
sqrt = np.sqrt
atan = np.arctan
acos = np.arccos 
pi = np.pi 
tpi = 2*pi 

def put_in_cell(x):
    while x < 0:
        x += 1
    while x >= 1:
        x -= 1  
    return x 

def read_hkl3(choice=None):
	cwd = os.getcwd()
	files = os.listdir(cwd)

	# read hkl
	if type(choice)!=str:
		print('read hkl into numpy arrays h k l F2 sigF2')
		print('Load hkl file:')
		hkl_files = {}
		i = -1
		for file in files:
			if file.endswith('.hkl'):
				i += 1
				hkl_files[i] = file  
				print(i, ": ", file)
	if choice is None:
		if i==0:
			choice = 0
		else:
			choice = input('choice 0: ')
			if choice:
				try:
					choice = int(choice)
				except:
					choice = 0
			else:
				choice = 0
	if type(choice)==int:
		filename = hkl_files[choice]
	else:
		filename = choice
	with open(filename,'r') as f:
		fread = f.read()
	lines = fread.split('\n')
	ha,ka,la,F2a,sigF2a = [],[],[],[],[]
	for line in lines:
		try:
			h,k,l,F2,sigF2 = line[:4],line[4:8],line[8:12],line[12:20],line[20:28]
			h,k,l,F2,sigF2 = int(h),int(k),int(l),float(F2),float(sigF2)
			if h or k or l:
				ha.append(h)
				ka.append(k)
				la.append(l)
				F2a.append(F2)
				sigF2a.append(sigF2)
		except:
			try:
				h,k,l,F2,sigF2,n = line[:4],line[4:8],line[8:12],line[12:20],line[20:28],line[28:]
				h,k,l,F2,sigF2,n = int(h),int(k),int(l),float(F2),float(sigF2),int(n)
				if h or k or l:
					ha.append(h)
					ka.append(k)
					la.append(l)
					F2a.append(F2)
					sigF2a.append(sigF2)
			except:
				pass
	ha = np.array(ha)
	ka = np.array(ka)
	la = np.array(la)
	F2a = np.array(F2a)
	sigF2a = np.array(sigF2a)
	return (ha,ka,la,F2a,sigF2a) 

def read_hkl2():
	cwd = os.getcwd()
	files = os.listdir(cwd)

	# read hkl
	print('\nLoad hkl file:')
	hkl_files = {}
	i = -1
	for file in files:
		if file.endswith('.hkl'):
			i += 1
			hkl_files[i] = file  
			print(i, ": ", file)
	if i==0:
		choice = 0
	else:
		choice = input('choice 0: ')
		if choice:
			try:
				choice = int(choice)
			except:
				choice = 0
		else:
			choice = 0
	filename = hkl_files[choice]
	with open(filename,'r') as f:
		fread = f.read()
	lines = fread.replace('\r','').split('\n')
	return lines

def read_hkl():
	print('read hkl file to make a dictionary of hkl F2 sigF2')
	lines = read_hkl2()
	hkl = {}
	for line in lines:
		try:
			h,k,l,F2,sigF2 = line.split()
			h,k,l,F2,sigF2 = int(h),int(k),int(l),float(F2),float(sigF2)
			if h or k or l:
				if (h,k,l) not in hkl:
					hkl[(h,k,l)]=[(F2,sigF2)]
				else:
					hkl[(h,k,l)].append((F2,sigF2))
		except:
			try:
				h,k,l,F2,sigF2,n = line.split()
				h,k,l,F2,sigF2,n = int(h),int(k),int(l),float(F2),float(sigF2),int(n)
				if h or k or l:
					if (h,k,l) not in hkl:
						hkl[(h,k,l)]=[(F2,sigF2,n)]
					else:
						hkl[(h,k,l)]=append((F2,sigF2,n))
			except:
				pass
	#print(hkl)
	return hkl 


def read_fcf():
	cwd = os.getcwd()
	files = os.listdir(cwd)

	# read fcf
	print('\nLoad fcf file:')
	fcf_files = {}
	i = -1
	for file in files:
		if file.endswith('.fcf'):
			i += 1
			fcf_files[i] = file  
			print(i, ": ", file)
	if i==0:
		choice = 0
	else:
		choice = input('choice 0: ')
		if choice:
			try:
				choice = int(choice)
			except:
				choice = 0
		else:
			choice = 0
	filename = fcf_files[choice]
	with open(filename,'r') as f:
		fread = f.read()
	lines = fread.replace('\r','').split('\n')
	fcf = {}
	for line in lines:
		try:
			h,k,l,F2,sigF2,F,Ph = line.split()
			h,k,l,F2,sigF2,F,Ph = int(h),int(k),int(l),float(F2),float(sigF2),float(F),float(Ph)
			fcf[(h,k,l)]=(F2,sigF2,F,Ph)
		except:
			pass
	#print(fcf)
	return fcf 


def get_cell(choice=None):
	cwd = os.getcwd()
	files = os.listdir(cwd)

	# read res
	if type(choice)!=str:
		print('get cell parameters from res file')
		print('\nLoad res file:')
		res_files = {}
		i = -1
		for file in files:
			if file.endswith('.res'):
				i += 1
				res_files[i] = file  
				print(i, ": ", file)
	if choice is None:
		if i==0:
			choice = 0
		else:
			choice = input('choice 0: ')
			if choice:
				try:
					choice = int(choice)
				except:
					choice = 0
			else:
				choice = 0
	if type(choice)==int:
		filename = res_files[choice]
	else:
		filename = choice
	with open(filename,'r') as f:
		fread = f.read()
	lines = fread.replace('\r','').split('\n')
	for line in lines:
		try:
			words = line.split()
			if words[0]=='CELL':
				lam = float(words[1])
				a = float(words[2])
				b = float(words[3])
				c = float(words[4])
				alfa = float(words[5])
				beta = float(words[6])
				gamma = float(words[7])
				return (lam,a,b,c,alfa,beta,gamma)
		except:
			pass

def read_res(choice=None):
	cwd = os.getcwd()
	files = os.listdir(cwd)

	# read res
	if type(choice)!=str:
		print('get cell parameters from res file')
		print('\nLoad res file:')
		res_files = {}
		i = -1
		for file in files:
			if file.endswith('.res'):
				i += 1
				res_files[i] = file  
				print(i, ": ", file)
	if choice is None:
		if i==0:
			choice = 0
		else:
			choice = input('choice 0: ')
			if choice:
				try:
					choice = int(choice)
				except:
					choice = 0
			else:
				choice = 0
	if type(choice)==int:
		filename = res_files[choice]
	else:
		filename = choice
	with open(filename,'r') as f:
		fread = f.read()
	lines = fread.replace('\r','').split('\n')
	return lines

def read_lst():
	cwd = os.getcwd()
	files = os.listdir(cwd)

	# read res
	print('\nLoad lst file:')
	lst_files = {}
	i = -1
	for file in files:
		if file.endswith('.lst'):
			i += 1
			lst_files[i] = file  
			print(i, ": ", file)
	if i==0:
		choice = 0
	else:
		choice = input('choice 0: ')
		if choice:
			try:
				choice = int(choice)
			except:
				choice = 0
		else:
			choice = 0
	filename = lst_files[choice]
	with open(filename,'r') as f:
		fread = f.read()
	lines = fread.replace('\r','').split('\n')
	return lines


def read_atoms(choice=None):
	cwd = os.getcwd()
	files = os.listdir(cwd)

	if type(choice)!=str:
		# read res
		print('read res to extract atom list: atom x y z')
		print('\nLoad res file:')
		res_files = {}
		i = -1
		for file in files:
			if file.endswith('.res'):
				i += 1
				res_files[i] = file  
				print(i, ": ", file)
	if choice is None:
		if i==0:
			choice = 0
		else:
			choice = input('choice 0: ')
			if choice:
				try:
					choice = int(choice)
				except:
					choice = 0
			else:
				choice = 0
	if type(choice)==int:
		filename = res_files[choice]
	else:
		filename = choice
	with open(filename,'r') as f:
		fread = f.read()
	lines = fread.split('\n')
	atoms = {}
	for line in lines:
		try:
			words = line.split()
			if words[0].upper()=='SFAC':
				for i in range(1,len(words)):
					atoms[str(i)] = words[i]
				break
		except:
			pass

	i, n = 0, len(lines)
	lines1 = []
	while 'LATT' not in lines[i].upper():
		lines1.append(lines[i])
		#print(lines[i])
		i += 1 
	lines2 = [lines[i]]
	#print(lines[i])
	i += 1
	while 'SYMM' in lines[i].upper():
		lines2.append(lines[i])
		#print(lines[i])
		i += 1
	lines3 = []
	while 'FVAR' not in lines[i].upper():
		lines3.append(lines[i])
		#print(lines[i])
		i += 1
	lines3.append(lines[i])
	#print(lines[i])
	i += 1
	lines4 = []
	while 'HKLF' not in lines[i].upper():
		lines4.append(lines[i])
		#print(lines[i])
		i += 1
	lines5 = []
	while i < n:
		lines5.append(lines[i])
		#print(lines[i])
		i += 1
	atomlist = []
	for line in lines4:
		try:
			if not line: continue
			if line.startswith(' '): continue
			words = line.split()
			if len(words)<5: continue
			x,y,z = float(words[2]),float(words[3]),float(words[4])
			#x,y,z = put_in_cell(x),put_in_cell(y),put_in_cell(z)
			atomlist.append((atoms[words[1]],words[1],x,y,z)) # atom, label, x,y,z
		except:
			pass
	return atomlist



def read_atoms2():
	cwd = os.getcwd()
	files = os.listdir(cwd)

	# read res
	print('read res to extract atom list: atom_label x y z')
	print('\nLoad res file:')
	res_files = {}
	i = -1
	for file in files:
		if file.endswith('.res'):
			i += 1
			res_files[i] = file  
			print(i, ": ", file)
	if i==0:
		choice = 0
	else:
		choice = input('choice 0: ')
		if choice:
			try:
				choice = int(choice)
			except:
				choice = 0
		else:
			choice = 0
	filename = res_files[choice]
	with open(filename,'r') as f:
		fread = f.read()
	lines = fread.split('\n')
	atoms = {}
	for line in lines:
		try:
			words = line.split()
			if words[0].upper()=='SFAC':
				for i in range(1,len(words)):
					atoms[str(i)] = words[i]
				break
		except:
			pass

	i, n = 0, len(lines)
	lines1 = []
	while 'LATT' not in lines[i].upper():
		lines1.append(lines[i])
		print(lines[i])
		i += 1 
	lines2 = [lines[i]]
	print(lines[i])
	i += 1
	while 'SYMM' in lines[i].upper():
		lines2.append(lines[i])
		print(lines[i])
		i += 1
	lines3 = []
	while 'FVAR' not in lines[i].upper():
		lines3.append(lines[i])
		print(lines[i])
		i += 1
	lines3.append(lines[i])
	print(lines[i])
	i += 1
	lines4 = []
	while 'HKLF' not in lines[i].upper():
		lines4.append(lines[i])
		print(lines[i])
		i += 1
	lines5 = []
	while i < n:
		lines5.append(lines[i])
		print(lines[i])
		i += 1
	atomlist = []
	for line in lines4:
		try:
			if not line: continue
			if line.startswith(' '): continue
			words = line.split()
			if len(words)<5: continue
			x,y,z = float(words[2]),float(words[3]),float(words[4])
			atomlist.append((words[0],x,y,z))
		except:
			pass
	return atomlist


def read_analyze_res(choice=None):
    print('read res file and analyze into parts')
    lines = read_res(choice)

    i, n = 0, len(lines)
    lines1 = []
    while 'LATT' not in lines[i].upper():
        lines1.append(lines[i])
        i += 1 
    lines2 = [lines[i]]
    i += 1
    while 'SYMM' in lines[i].upper():
        lines2.append(lines[i])
        i += 1
    lines3 = []
    while 'FVAR' not in lines[i].upper():
        lines3.append(lines[i])
        i += 1
    lines3.append(lines[i])
    i += 1
    lines4 = []
    while 'HKLF' not in lines[i].upper():
        lines4.append(lines[i])
        i += 1
    lines5 = []
    while i < n:
        lines5.append(lines[i])
        i += 1
    return (lines1,lines2,lines3,lines4,lines5)


def read_analyze_res2(choice=None):
    #print('read res file and analyze into parts')
    lines = read_res(choice)

    i, n = 0, len(lines)
    lines1 = []
    while 'LATT' not in lines[i].upper():
        lines1.append(lines[i])
        i += 1 
    lines2 = [lines[i]]
    i += 1
    while 'SYMM' in lines[i].upper():
        lines2.append(lines[i])
        i += 1
    lines3 = []
    while 'FVAR' not in lines[i].upper():
        lines3.append(lines[i])
        i += 1
    lines3.append(lines[i])
    i += 1
    lines4 = []
    while 'HKLF' not in lines[i].upper():
        lines4.append(lines[i])
        i += 1
    lines5 = []
    while i < n:
        lines5.append(lines[i])
        i += 1

    for line in lines1:
        try:
            words = line.split()
            if words[0]=='CELL':
                lam = float(words[1])
                a = float(words[2])
                b = float(words[3])
                c = float(words[4])
                alfa = float(words[5])
                beta = float(words[6])
                gamma = float(words[7])
                break
        except:
            pass
    alfa = radians(alfa)
    beta = radians(beta)
    gamma = radians(gamma)

    A = np.array([
    [a*a,a*b*cos(gamma),a*c*cos(beta)],
    [a*b*cos(gamma),b*b,b*c*cos(alfa)],
    [a*c*cos(beta),b*c*cos(alfa),c*c]
    ])

    res_parts = (lines1,lines2,lines3,lines4,lines5)
    return (res_parts,A)

