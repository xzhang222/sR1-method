import sys
import time 
from tools import *

A = matrix_A('a.res')

with open('a.res','r') as f:
	text=f.read()
lines=text.split('\n')
i_FVAR,i_HKLF=0,0
for i in range(len(lines)):
	if lines[i].startswith('FVAR'): i_FVAR=i  
	if lines[i].startswith('HKLF'):i_HKLF=i 
res_start_lines=lines[:i_FVAR+1]
res_end_lines=lines[i_HKLF:]

with open('history.txt','r') as f:
	text=f.read()
lines=text.split('\n')

atom_list=[]
model_lines=[]
is_model=0
i=-1
for line in lines[:-1]:
	i+=1
	if i<000: continue
	words=line.split()
	try:
		if words[5]=='11.00' and words[6]=='0.05':
			is_model=1
		else:
			is_model=0
	except:
		is_model=0
	if is_model:
		model_lines.append(lines[i])
		if not atom_list: print('Find model at line ',i,' of ',len(lines),'\n')
		atom,label,x,y,z=words[0],words[1],float(words[2]),float(words[3]),float(words[4])
		atom=''
		for letter in words[0]:
			if letter.isalpha(): atom+=letter
		atom_list.append((atom,label,x,y,z))
	else:
		if atom_list:
			res_lines=res_start_lines+model_lines+['','']+res_end_lines
			with open('a_view.res','w') as f:
				for l in res_lines:
					print(l,file=f)
			display_structure(atom_list,A)
			atom_list=[]
			model_lines=[]
print('done')
