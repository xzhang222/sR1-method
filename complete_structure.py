import sys
import time 
from tools import *
import numpy
from read_data import read_atoms

starttime=time.time()
tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
with open(sys.argv[0],'r') as f:
    text=f.read()
with open('history.txt','a') as f:
    print('\n\nstep: \n',tt,file=f)
    print('\n\nprogram '+sys.argv[0]+':\n'+text+'\n\n',file=f)

A=matrix_A('a.res')
atom_list=read_atoms('a.res')
atoms,labels,s=atomj_solution(atom_list)
for i in range(len(s)):
	s[i]=numpy.array(s[i])
if 0:
    for sc1,sc2,sx in [(46,76,81),]:
        x,y,z=s[sc1-1]+s[sc2-1]-s[sx-1]
        atom_list.append(('C','1',x,y,z))
if 0:
    C=(s[23-1]+s[67-1])/2
    for i in range(102,115):
        X=s[i-1]
        x,y,z=2*C-X
        if i==102:
            atom_list.append(('N','1',x,y,z))
        else:
            atom_list.append(('C','1',x,y,z))
if 0:
    for i,j in [(2,8),(4,10)]:
        A,B=s[i-1],s[j-1]
        C,D=((2*A+B)/3,(A+2*B)/3)
        for X in [C,D]:
            x,y,z=X
            atom_list.append(('S','1',x,y,z))
if 0:
    for i1,i2,i3,i4 in [(2,11,4,13),(12,8,14,10),(11,13,12,14)]:
        A1,B,C,D=s[i1-1],s[i2-1],s[i3-1],s[i4-1]
        P,Q=(A1+B)/2,(C+D)/2
        U=(P-Q)/length(P,Q,A)
        O=(P+Q)/2
        r=1.30
        x,y,z=O+r*U/2
        atom_list.append(('C','1',x,y,z))
        x,y,z=O-r*U/2 
        atom_list.append(('C','1',x,y,z))

save_history(atom_list,runs='complete structure')
print('all done')
