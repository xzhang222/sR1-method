import sys
import time 
from tools import *

starttime=time.time()
tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
with open(sys.argv[0],'r') as f:
    text=f.read()
with open('history.txt','a') as f:
    print('\n\nstep: \n',tt,file=f)
    print('\n\nprogram '+sys.argv[0]+':\n'+text+'\n\n',file=f)

atom_list = read_atoms('correct.res')
A = matrix_A('a.res')

if 1:
	for i in range(len(atom_list)):
		atom,label,x,y,z=atom_list[i]
		atom_list[i]=atom,label,1-x,1-y,1-z

atom,label,x0,y0,z0=atom_list[2]
dx,dy,dz=0.3-x0,0.3-y0,0.3-z0
for i in range(len(atom_list)):
	atom,label,x,y,z=atom_list[i]
	atom_list[i]=atom,label,x+dx,y+dy,z+dz

with open('peak_list.txt','r') as f:
	text=f.read()
lines=text.split('\n')

peak_list=[]
for line in lines:
	try:
		atom,label,x,y,z,hh=line.split()
		x,y,z,hh=float(x),float(y),float(z),float(hh)
		peak_list.append((atom,label,x,y,z,hh))
	except:
		pass

with open('a.res','r') as f:
    text=f.read()
lines=text.split('\n')
i_FVAR,i_HKLF=0,0
for i in range(len(lines)):
    if lines[i].startswith('FVAR'): i_FVAR=i  
    if lines[i].startswith('HKLF'):i_HKLF=i 
res_start_lines=lines[:i_FVAR+1]
res_end_lines=lines[i_HKLF:]

model_lines=[]
for j in range(len(atom_list)):
	atom,label,x0,y0,z0=atom_list[j]
	dmin=1e200
	for i in range(len(peak_list)):
		am,lb,x,y,z,hh = peak_list[i]
		d=d_min5((x0,y0,z0),(x,y,z),A)
		if d<dmin:
			dmin=d 
			line=atom+str(j+1)+' '+label+' '+str(x)+' '+str(y)+' '+str(z)+' '+' 11.00 '+str(hh)
	model_lines.append(line)

res_lines=res_start_lines+model_lines+['','']+res_end_lines
with open('peak_view.res','w') as f:
    for l in res_lines:
        print(l,file=f)

with open('history.txt','a') as f:
    print('',file=f)
    print('',file=f)
    for l in res_lines:
        print(l,file=f)
    print('',file=f)
    print('',file=f)


print('done')
