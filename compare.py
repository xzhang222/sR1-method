import sys
import time 
import numpy
from tools import *

with open(sys.argv[0],'r') as f:
    text=f.read()
with open('history.txt','a') as f:
    print('\n\nprogram '+sys.argv[0]+':\n'+text+'\n\n',file=f)

if 0:
    re_order()   

if 0:
    generate_random_model()   

if 0:
    atom_list = read_atoms('a.res')
    save_history(atom_list,runs='current model')   

if 0:
    invert_model(res='a.res')

if 0:
    re_arrange() 

if 0:
    pairing2(correct_res='correct.res',res='a.res', s=0.5)

if 0: # molecular model
    atom_list=read_atoms('a.res')
    #save_history(atom_list,runs='starting model')
    atoms,labels,s=atomj_solution(atom_list)
    for i in range(len(s)):
        s[i]=numpy.array(s[i])
    p1,p2,p3=s[0],s[6],s[1]
    #p3=numpy.array([0.3,0.3,0.3])
    A = matrix_A('a.res')
    xp,yp,zp=local_xpypzp(p1,p2,p3,A)
    for i in range(len(s)):
        s[i]=cell_to_local_cartesian(s[i],p1,xp,yp,zp,A)
    atom_list=[]
    for i in range(len(s)):
        x,y,z=s[i]
        atom_list.append((atoms[i],labels[i],x,y,z))
    with open('MoO4SiC16_i.txt','w') as f:
        for a,l,x,y,z in atom_list:
            print(a,l,x,y,z,file=f)
            #print(a,l,-x,-y,-z,file=f)

if 0:  # generate a fragment
    #fragment0,n_fold=make_molecule('OC4.txt')
    fragment0,n_fold=make_benzene()
    p0=(0.3,0.3,0.3)
    #p0=(0,0,0)
    psi,phi,ita=0,0,0
    A = matrix_A('a.res')
    C,D=getCD(A)
    atom_list=add_fragment(p0,psi,phi,ita,D,fragment0)
    save_history(atom_list,'one fragment',True)

if 0: # find orientation
    atom_list=read_atoms('a.res')
    save_history(atom_list,runs='starting model')
    atoms,labels,s=atomj_solution(atom_list)
    for i in range(len(s)):
        s[i]=numpy.array(s[i])
    p1,p2,p3=s[27],s[6],s[29]
    #p3=numpy.array([0.3,0.3,0.3])
    A = matrix_A('a.res')
    xp,yp,zp=local_xpypzp(p1,p2,p3,A)
    xpc,ypc,zpc=xpypzp(A) # cell cartesian
    c11,c21,c31=to_cartesian_coord(xp,xpc,ypc,zpc,A)
    c12,c22,c32=to_cartesian_coord(yp,xpc,ypc,zpc,A)
    c13,c23,c33=to_cartesian_coord(zp,xpc,ypc,zpc,A)
    C=numpy.array([[c11,c12,c13],[c21,c22,c23],[c31,c32,c33]])
    print(C)
    #C=numpy.linalg.inv(C)
    #C=C.T
    #print(C)
    phi=acos(C[0,0])
    if C[2,0]>0:
        psi=acos(C[1,0]/sin(phi))
    else:
        psi=tpi-acos(C[1,0]/sin(phi))
    Psi=numpy.array([[1,0,0],[0,cos(psi),sin(psi)],[0,-sin(psi),cos(psi)]])
    Phi=numpy.array([[cos(phi),sin(phi),0],[-sin(phi),cos(phi),0],[0,0,1]])
    B=Phi@Psi@C
    print(B)
    if B[2,1]>0:
        ita=acos(B[1,1])
    else:
        ita=tpi-acos(B[1,1])
    psi,phi,ita=degrees(psi),degrees(phi),degrees(ita)
    with open('history.txt','a') as f:
        print('\n\norientation found:\n',file=f)
        print(psi,phi,ita,1,file=f)

if 0:  # search orientation
    x0,y0,z0=245.95,75.6,192.31
    x0,y0,z0=114.05,104.4,12.3
    with open('orientations.txt','r') as f:
        text=f.read()
    lines=text.split('\n')[:-1]

    def get_orientation(i):
        psi,phi,ita,ff=lines[i].split()
        psi,phi,ita,ff=float(psi),float(phi),float(ita),float(ff)
        return (psi,phi,ita)

    dmin,imin=1e8,0
    for i in range(len(lines)):
        x,y,z=get_orientation(i)
        d=abs(x-x0)+abs(y-y0)+abs(z-z0)
        if d<dmin:
            dmin,imin=d,i 
    print(imin,get_orientation(i),' to match ',(x0,y0,z0))

    
with open('history.txt','a') as f:
    print('\n\n\nstep:  \n\n\n',file=f)
print('done')

