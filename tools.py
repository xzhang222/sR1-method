from math import radians, degrees, log 
import math 
from read_data import get_cell,  read_hkl3, read_atoms
import numpy
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import dblquad
import sys
from random import sample, choice, random  
import time
from pyperclip import copy as cp, paste as pst 
from elements3 import elements
from scipy.fft import fftn, ifftn 
import pp  # this module is available from parallelpython.com

job_server = pp.Server(ppservers=())
ncpus = job_server.get_ncpus()

label_dic={}
try:
    with open('a.res','r') as f:
        text=f.read()
    lines=text.split('\n')
    i_FVAR,i_HKLF=0,0
    for i in range(len(lines)):
        if lines[i].startswith('FVAR'): i_FVAR=i  
        if lines[i].startswith('HKLF'):i_HKLF=i 
        if lines[i].startswith('SFAC'):
            words=lines[i].split()
            for j in range(1,len(words)):
                label_dic[words[j]]=str(j)
    res_start_lines=lines[:i_FVAR+1]
    res_end_lines=lines[i_HKLF:]
except:
    res_start_lines=[]
    res_end_lines=[]


do_pauss=False   
delay=0.1

sin = numpy.sin  
cos = numpy.cos 
exp = numpy.exp 
sqrt = numpy.sqrt
atan = numpy.arctan
acos = numpy.arccos 
asin = numpy.arcsin
pi = numpy.pi 
tpi = 2*pi 
fpi = 4*pi 

NX, NY, NZ = None, None, None  
xg, yg, zg = None, None, None 
Z_atoms = None  
SIN, COS = None, None  
f2a, sl = None, None 

the_heavy = ['Mo','Se','I','Pd','Re','Pt','Sn','W','Br'] # dd=2.2 for the heavy, otherwise 1.2 
the_light = ['C','N','O','F']
rlight, rheavy = 0.7, 1.25   # covalent radii
r_covalent={'C':0.770,'N':0.700,'O':0.660,'F':0.640,'H':0.320,
            'S':1.030,'P':1.100,'Cl':0.990,'Si':1.170,'Cu':1.280,
            'I':1.330,'Pt':1.370,'Mo':1.360,'Se':1.170,'Br':1.140,
            'K':2.270,'Na':1.860,'Cs':2.650,'Rb':2.480,'Ag':1.440,
            'Ni':1.250,'Pd':1.380,'Re':1.370,'Sn':1.400,}
colors={'C':'dimgrey','N':'cornflowerblue','O':'red','F':'lawngreen','H':'ivory',
        'S':'yellow','P':'violet','Cl':'green','Si':'silver','Cu':'peru',
        'I':'magenta','Pt':'darkgrey','Mo':'lightgrey','Se':'darkviolet','Br':'orange',
        'K':'gainsboro','Na':'gainsboro','Cs':'gainsboro','Rb':'gainsboro','Ag':'silver',
        'Ni':'yellowgreen','Pd':'azure','Re':'lightcyan','Sn':'lightcyan',}
defaultcolor='lightsteelblue'

def cc(aa,bb,gamma):
    # gamma in degrees
    gamma=radians(gamma)
    return sqrt(aa*aa+bb*bb-2*aa*bb*cos(gamma))

def put_in_cell(x):
    while x < 0:
        x += 1
    while x >= 1:
        x -= 1  
    return x 

def st1(x):
    r = str(x)
    n = 4-len(r)
    if n>0:
        r = ' '*n + r 
    return r 

def st2(x):
    r = str(x)
    n = 8-len(r)
    if n>0:
        r = ' '*n + r 
    return r 

def st3(x):
    r = str(x)
    n = 6-len(r)
    if n>0:
        r += ' '*n 
    return r 


def matrix_A(choice=None):
    # choice = int 0,1,2... or str filename
    lam,a,b,c,alfa,beta,gamma = get_cell(choice)
    #print(lam,a,b,c,alfa,beta,gamma)
    alfa = radians(alfa)
    beta = radians(beta)
    gamma = radians(gamma)

    A = numpy.array([
    [a*a,a*b*cos(gamma),a*c*cos(beta)],
    [a*b*cos(gamma),b*b,b*c*cos(alfa)],
    [a*c*cos(beta),b*c*cos(alfa),c*c]
    ])
    return A 

def abc(A):
    a,b,c = numpy.sqrt(A[0,0]),numpy.sqrt(A[1,1]),numpy.sqrt(A[2,2])
    return (a,b,c)

def alfabetagamma(A):
    a,b,c = abc(A)
    alfa,beta,gamma = A[1,2]/b/c,A[0,2]/a/c,A[0,1]/a/b
    alfa,beta,gamma =  acos(alfa),acos(beta),acos(gamma)
    alfa,beta,gamma = degrees(alfa),degrees(beta),degrees(gamma)
    return (alfa,beta,gamma)

def NaNbNc(A):
    # cut a into Na parts, b into Nb parts, c into Nc parts
    # cut cell into Na*Nb*Nc boxes
    # the distance between opposite faces of the boxes are 0.5 A
    alfa,beta,gamma=alfabetagamma(A)
    alfa,beta,gamma=radians(alfa),radians(beta),radians(gamma)
    ca,cb,cg=cos(alfa),cos(beta),cos(gamma)
    f=sqrt(1-ca*ca-cb*cb-cg*cg+2*ca*cb*cg)
    a,b,c=abc(A)
    s=0.5
    sa,sb,sg=sin(alfa),sin(beta),sin(gamma)
    Na,Nb,Nc=int(a*f/s/sa),int(b*f/s/sb),int(c*f/s/sg)
    return (Na,Nb,Nc)

def matrix_inv(A):
    # inverse of A
    return numpy.linalg.inv(A)


def correct(dx,x2):
    while dx<=-0.5:
        dx+=1
        x2+=1
    while dx>0.5:
        dx-=1
        x2-=1
    return (dx,x2)

def d_min(p1,p2,A,symmetries=['x,y,z']):
    # apply some symmetry and additional lattice shifts
    # to move p2 to p2' such that distance between p1 and p2'
    # reaches d_min
    # returns d_min and p2'
    x1,y1,z1=p1
    p2p,dmin=None,1e100
    for symmetry in symmetries:
        x,y,z=p2 
        x2,y2,z2=eval(symmetry)
        dx,dy,dz=x2-x1,y2-y1,z2-z1
        dx,x2=correct(dx,x2)
        dy,y2=correct(dy,y2)
        dz,z2=correct(dz,z2)
        d=d_exact((dx,dy,dz),A)
        if d<dmin:
            dmin=d  
            p2p=(x2,y2,z2)
    return (dmin,numpy.array(p2p))

def d_min2(p1,p2,A):
    # apply lattice shifts to p2 (no symmetries applied)
    # to move p2 to p2' such that distance between p1 and p2'
    # reaches d_min
    # returns d_min and p2'
    x1,y1,z1=p1
    x2,y2,z2=p2  
    dx,dy,dz=x2-x1,y2-y1,z2-z1
    dx,x2=correct(dx,x2)
    dy,y2=correct(dy,y2)
    dz,z2=correct(dz,z2)
    d=d_exact((dx,dy,dz),A)
    p2p=(x2,y2,z2)
    return (d,numpy.array(p2p))

def d_min3(p1,p2,A):
    # apply lattice shifts to p2
    # to move p2 to p2' such that distance between p1 and p2'
    # reaches d_min
    # only returns d_min^2
    x1,y1,z1=p1
    x2,y2,z2=p2  
    dx,dy,dz=x2-x1,y2-y1,z2-z1
    dx,x2=correct(dx,x2)
    dy,y2=correct(dy,y2)
    dz,z2=correct(dz,z2)
    return dis_exact((dx,dy,dz),A)

def d_min4(p1,p2,A):
    # apply lattice shifts to p2
    # to move p2 to p2' such that distance between p1 and p2'
    # reaches d_min
    # returns d_min^2 and p2'
    x1,y1,z1=p1
    x2,y2,z2=p2  
    dx,dy,dz=x2-x1,y2-y1,z2-z1
    dx,x2=correct(dx,x2)
    dy,y2=correct(dy,y2)
    dz,z2=correct(dz,z2)
    d=dis_exact((dx,dy,dz),A)
    p2p=(x2,y2,z2)
    return (d,numpy.array(p2p))

def d_min5(p1,p2,A):
    # apply lattice shifts to p2
    # to move p2 to p2' such that distance between p1 and p2'
    # reaches d_min
    # returns d_min
    return math.sqrt(d_min3(p1,p2,A))

def d_exact(p,A):
    return math.sqrt(dis_exact(p,A))

def dis_exact(p,A): # exact
    # returns |p|^2
    p1 = numpy.array(p)
    return numpy.matmul(p1,numpy.matmul(A,p1[:,numpy.newaxis]))[0]

def dis(p,A): # shortest
    # returns |p|^2
    return d_min3(p,[0,0,0],A) # this method has bug (when cell angles deviate from 90 degrees)
                               # but for excluding ghost peaks, dmin need not be accurate
                               # correcting this bug will make calculation unneccessarily slower
                               # put atoms in order may occasionally incorrect because of this bug
                               # just use unique command to make correction

def length(p1,p2,A): # return actual length, not shortest length
    p1 = numpy.array(p1)
    p2 = numpy.array(p2)
    p = p1-p2
    return sqrt(numpy.matmul(p,numpy.matmul(A,p[:,numpy.newaxis]))[0])

def dot(p1,p2,A):
    p1 = numpy.array(p1)
    p2 = numpy.array(p2)
    return numpy.matmul(p1,numpy.matmul(A,p2[:,numpy.newaxis]))[0]

def cross(p1,p2,A):
    x1,y1,z1 = p1
    x2,y2,z2 = p2 
    T = matrix_inv(A)
    cell_V = sqrt(numpy.linalg.det(A))

    Z =numpy.array( [y1*z2-z1*y2,z1*x2-x1*z2,x1*y2-y1*x2] )*cell_V #base a*,b*,c*
    Z = numpy.matmul(T,Z[:,numpy.newaxis])  # switch to base a, b, c
    Z = Z.reshape(3)
    return Z

def xpypzp(A):
    # set up cell cartesian unit vectors xp, yp, zp 
    # as linear combination of cell base vectors a, b, c
    # xp in direction of a
    # yp in direction of component of b perpendicular to a
    # zp = xp x yp
    a,b,c=abc(A)

    # convert to cartesian coordinates
    pa=numpy.array([1.,0.,0.])
    pb=numpy.array([0.,1.,0.])
    pc=numpy.array([0.,0.,1.])
    xp=pa/a  
    bx=xp*dot(pb,xp,A)
    by=pb-bx
    yp=by/length(by,[0,0,0],A)
    zp=cross(xp,yp,A)
    return (xp,yp,zp)

def to_cartesian_coord(p,xp,yp,zp,A):
    x=dot(p,xp,A)
    y=dot(p,yp,A)
    z=dot(p,zp,A)
    return numpy.array([x,y,z])

def to_cell_coord(p,xp,yp,zp):
    x,y,z=p  
    return x*xp+y*yp+z*zp  


def local_xpypzp(p1,p2,p3,A):
    # set up local cartesian unit vectors xp, yp, zp 
    # as linear combination of cell base vectors a, b, c
    # xp in direction of p1p2
    # yp in direction of component of p1p3 perpendicular to p1p2
    # zp = xp x yp
    pa=p2-p1
    pb=p3-p1
    xp=pa/length(pa,[0,0,0],A)
    bx=xp*dot(pb,xp,A)
    by=pb-bx
    yp=by/length(by,[0,0,0],A)
    zp=cross(xp,yp,A)
    return (xp,yp,zp)

def cell_to_local_cartesian(p,p1,xp,yp,zp,A):
    pp=p-p1
    x=dot(pp,xp,A)
    y=dot(pp,yp,A)
    z=dot(pp,zp,A)
    return numpy.array([x,y,z])

def cells_to_local_cartesians(ps,p1,xp,yp,zp,A):
    return [cell_to_local_cartesian(p,p1,xp,yp,zp,A) for p in ps]

def local_cartesian_to_cell(pc,p1,xp,yp,zp):
    x,y,z=pc  
    return x*xp+y*yp+z*zp+p1  

def local_cartesians_to_cells(pcs,p1,xp,yp,zp):
    return [p[0]*xp+p[1]*yp+p[2]*zp+p1 for p in pcs]  

def getCD(A):
    a,b,c=abc(A)
    alfa,beta,gamma=alfabetagamma(A)
    alfa = radians(alfa)
    beta = radians(beta)
    gamma = radians(gamma)
    C11,C21,C31=a,0.0,0.0
    C12,C22,C32=b*cos(gamma),b*sin(gamma),0.0 
    C13=c*cos(beta)
    C23=c*(cos(alfa)-cos(beta)*cos(gamma))/sin(gamma)
    V=cell_volume(A)
    C33=V/a/b/sin(gamma)
    C=numpy.array([[C11,C12,C13],[C21,C22,C23],[C31,C32,C33]])
    return (C,numpy.linalg.inv(C))

def to_cartesian(p,C):
    return numpy.matmul(C,p[:,numpy.newaxis]).reshape(3)

def to_cell(p,D):
    return numpy.matmul(D,p[:,numpy.newaxis]).reshape(3)

def to_cartesians(solution,C):
    return [to_cartesian(numpy.array(p),C) for p in solution]

def to_cells(csolution,D):
    return [to_cell(numpy.array(p),D) for p in csolution]

def Wyz(psi):
    # angle in radians
    c,s=numpy.cos(psi),numpy.sin(psi)
    return numpy.array([[1.0,0.0,0.0],[0.0,c,-s],[0.0,s,c]])

def Wxy(phi):
    # angle in radiants
    c,s=numpy.cos(phi),numpy.sin(phi)
    return numpy.array([[c,-s,0.0],[s,c,0.0],[0.0,0.0,1.0]])

def getW(psi,phi,ita):
    # angles are in degrees
    # in cartesian coordinates
    # the point original located at p
    # after rotating by three angles, it reaches W*p
    # consider the point is fixed in a rotating frame
    # first rotate the frame by angle psi from y to z
    # next rotate the frame by angle phi from x to y
    # finally rotate the frame by angle ita from y to z 
    psi,phi,ita=math.radians(psi),math.radians(phi),math.radians(ita)
    W1,W2,W3=Wyz(psi),Wxy(phi),Wyz(ita)
    return numpy.matmul(W1,numpy.matmul(W2,W3))

def rotate(p,W):
    return numpy.matmul(W,p[:,numpy.newaxis]).reshape(3)

def rotates(csolution,W):
    return [rotate(numpy.array(p),W) for p in csolution]

def make_molecule(molecule_file):
    with open(molecule_file,'r') as f:
        text=f.read()
    lines=text.split('\n')
    atoms,labels,ps,n_fold=[],[],[],1
    for line in lines:
        try:
            atom,label,x,y,z=line.split()
            x,y,z=float(x),float(y),float(z)
            atoms.append(atom)
            labels.append(label)
            ps.append(numpy.array([x,y,z]))
        except:
            try:
                n_fold=line.strip()
                n_fold=int(n_fold)
            except:
                pass
    # p0=numpy.zeros(3)
    # Ztot=0
    # for i in range(len(atoms)):
    #     Z=elements[atoms[i]]['Z']
    #     Ztot+=Z 
    #     p0=Z*ps[i]
    # p0/=Ztot
    # for i in range(len(ps)):
    #     ps[i]-=p0
    molecule=[]
    for i in range(len(ps)): 
        atom=atoms[i]
        label=labels[i]
        x,y,z=ps[i]
        molecule.append((atom,label,x,y,z))
    return (molecule,n_fold) 

def make_invert_molecule():
    with open('molecule.txt','r') as f:
        text=f.read()
    lines=text.split('\n')
    atoms,labels,ps,n_fold=[],[],[],1
    for line in lines:
        try:
            atom,label,x,y,z=line.split()
            x,y,z=float(x),float(y),float(z)
            atoms.append(atom)
            labels.append(label)
            ps.append(numpy.array([x,y,z]))
        except:
            try:
                n_fold=line.strip()
                n_fold=int(n_fold)
            except:
                pass
    # p0=numpy.zeros(3)
    # Ztot=0
    # for i in range(len(atoms)):
    #     Z=elements[atoms[i]]['Z']
    #     Ztot+=Z 
    #     p0=Z*ps[i]
    # p0/=Ztot
    # for i in range(len(ps)):
    #     ps[i]-=p0
    molecule=[]
    for i in range(len(ps)): 
        atom=atoms[i]
        label=labels[i]
        x,y,z=ps[i]
        molecule.append((atom,label,-x,-y,-z))
    return molecule 

def make_linear(molecule_file):
    with open(molecule_file,'r') as f:
        text=f.read()
    lines=text.split('\n')
    molecule=[]
    for line in lines:
        try:
            atom,label,r=line.split()
            r=float(r)
            molecule.append((atom,label,r,0,0))
        except:
            pass
    return molecule 


def make_benzene():
    # 6-fold rotation axis along x-axis
    n_fold=6
    r=1.39 
    benzene=[]
    for i in range(6):
        th=radians(60.0*i)
        y,z=r*cos(th),r*sin(th)
        benzene.append(('C','1',0.0,y,z))
    return (benzene,n_fold) 

def make_C3_sp2():
    # 2-fold rotation axis along x-axis
    n_fold=2
    r=1.39
    fragment=[]
    th=radians(60.0)
    x,y,z=r*cos(th),r*sin(th),0.0
    fragment.append(('C','1',z,z,z))
    fragment.append(('C','1',x,y,z))
    fragment.append(('C','1',x,-y,z))
    return (fragment,n_fold)

def make_C3_sp3():
    # 2-fold rotation axis along x-axis
    n_fold=2
    r=1.52
    fragment=[]
    th=radians(109.5/2)
    x,y,z=r*cos(th),r*sin(th),0.0
    fragment.append(('C','1',z,z,z))
    fragment.append(('C','1',x,y,z))
    fragment.append(('C','1',x,-y,z))
    return (fragment,n_fold)

def make_CNC():
    n_fold=1
    r=1.33
    fragment=[]
    th=radians(122.0)
    x,y,z=r*cos(th),r*sin(th),0.0
    fragment.append(('N','1',z,z,z))
    fragment.append(('C','1',x,y,z))
    fragment.append(('C','1',1.45,z,z))
    return (fragment,n_fold)

def make_CCON():
    n_fold=1
    r1,r2,r3=1.52,1.33,1.23
    fragment=[]
    th1,th2=radians(121.0),radians(-123.0)
    x1,y1,z=r1*cos(th1),r1*sin(th1),0.0
    x2,y2=r2*cos(th2),r2*sin(th2)
    fragment.append(('C','1',z,z,z))
    fragment.append(('O','1',r3,z,z))
    fragment.append(('C','1',x1,y1,z))
    fragment.append(('N','1',x2,y2,z))
    return (fragment,n_fold)

def make_SS():
    # inf-fold rotation axis along x-axis
    n_fold=10
    r=2.034
    fragment=[]
    x,y,z=r,0.0,0.0
    fragment.append(('S','1',y,y,z))
    fragment.append(('S','1',x,y,z))
    return (fragment,n_fold)

def make_SC():
    # inf-fold rotation axis along x-axis
    n_fold=10
    r=1.8
    fragment=[]
    x,y,z=r,0.0,0.0
    fragment.append(('S','1',y,y,z))
    fragment.append(('C','1',x,y,z))
    return (fragment,n_fold)


def make_benzene_tip():
    # 2-fold rotation axis along x-axis
    n_fold=2
    r=1.39
    fragment=[]
    th=radians(60.0)
    x,y,z=r*cos(th),r*sin(th),0.0
    fragment.append(('C','1',z,z,z))
    fragment.append(('C','1',x,y,z))
    fragment.append(('C','1',x+r,y,z))
    fragment.append(('C','1',2*x+r,z,z))
    fragment.append(('C','1',x+r,-y,z))
    fragment.append(('C','1',x,-y,z))
    return (fragment,n_fold)

def make_benzenestar():
    # 6-fold rotation axis along x-axis
    n_fold=6
    r=1.39 
    benzene=[]
    for i in range(6):
        th=radians(60.0*i)
        y,z=r*cos(th),r*sin(th)
        benzene.append(('C','1',0.0,y,z))
    for i in range(6):
        th=radians(60.0*i)
        y,z=2*r*cos(th),2*r*sin(th)
        benzene.append(('C','1',0.0,y,z))
    return (benzene,n_fold) 

def make_S2():
    b=2.034 
    frag=[]
    for r in [b/2,-b/2]:
        frag.append(('S','3',r,0,0))
    return frag 


def make_ethynylbenzene():
    # 2-fold rotation axis along x-axis
    n_fold=2
    r=1.39 
    ethynylbenzene=[]
    for i in range(6):
        th=radians(60.0*i)
        x,y,z=r*cos(th),r*sin(th),0.0
        ethynylbenzene.append(('C','1',x,y,z))
    d=1.3
    ethynylbenzene.append(('C','1',r+r,0.0,0.0))
    ethynylbenzene.append(('C','1',r+r+d,0.0,0.0))
    return (ethynylbenzene,n_fold) 


def make_PF6():
    # 4-fold rotation axis along x-axis
    n_fold=4
    r=1.59 
    PF6=[]
    PF6.append(('P','5',0.0,0.0,0.0))
    PF6.append(('F','6',r,0.0,0.0))
    PF6.append(('F','6',-r,0.0,0.0))
    PF6.append(('F','6',0.0,r,0.0))
    PF6.append(('F','6',0.0,-r,0.0))
    PF6.append(('F','6',0.0,0.0,r))
    PF6.append(('F','6',0.0,0.0,-r))
    return (PF6,n_fold) 

def xy(r,th):
    # th in degrees
    th=radians(th)
    x,y=r*cos(th),r*sin(th)
    return (x,y)

def make_pentagon():
    # 5-fold rotation axis along x-axis
    n_fold=5
    L=1.34
    th=radians(36)
    r=L/2/sin(th)
    pentagon=[]
    pentagon.append(('C','1',0.0,r,0.0))
    x,y=xy(r,72)
    pentagon.append(('C','1',0.0,x,y))
    x,y=xy(r,72*2)
    pentagon.append(('C','1',0.0,x,y))
    x,y=xy(r,72*3)
    pentagon.append(('C','1',0.0,x,y))
    x,y=xy(r,72*4)
    pentagon.append(('C','1',0.0,x,y))
    return (pentagon,n_fold)

def add_fragment(p,psi,phi,ita,D,fragment0):
    # benzene0=make_benzene()
    # p in cell coordinate
    # angles in degrees
    # D to convert cartesian to cell coord
    benzene0=[numpy.array([f[2],f[3],f[4]]) for f in fragment0]
    W=getW(psi,phi,ita)
    p0=numpy.array(p)
    benzene1=rotates(benzene0,W)
    benzene2=to_cells(benzene1,D)
    benzene3=[p+p0 for p in benzene2]
    atom_list=[]
    for i in range(len(benzene3)):
        atom,label,x,y,z=fragment0[i]
        x,y,z=benzene3[i]
        atom_list.append((atom,label,x,y,z))
    return atom_list

def add_sym_fragment(p,atom_list):
    # p in cell fraction coordinate, it is of atom 3
    # atom_list: known partial model
    # P2(1) space group
    atom1,label1,x1,y1,z1=atom_list[-2]
    atom2,label2,x2,y2,z2=atom_list[-1]
    x3,y3,z3=p  
    x4,y4,z4=x3+x1-x2,y3-y1+y2,z3+z1-z2
    return [(atom1,label1,x3,y3,z3),(atom2,label2,x4,y4,z4)]


def add_linear(p,theta,phi,D,fragment0):
    # fragment0: atom, label, r,0,0
    # p in cell coordinate
    # angles in degrees
    # D to convert cartesian to cell coord
    k=3.14159265/180  
    theta*=k  
    phi*=k  
    x0,y0,z0=p 
    atom_list=[]
    for atom,label,r,y,z in fragment0:
        rp=r*numpy.sin(theta)
        x,y,z=rp*numpy.cos(phi),rp*numpy.sin(phi),r*numpy.cos(theta)
        x,y,z=to_cell(numpy.array([x,y,z]),D)
        atom_list.append((atom,label,x0+x,y0+y,z0+z))
    return atom_list

def d_cartesian(p1,p2):
    p=p2-p1
    return sqrt((p*p).sum())

def to_cartesian_solution(solution,xp,yp,zp,A):
    cart_solution=[]
    for p in solution:
        pc=to_cartesian_coord(p,xp,yp,zp,A)
        cart_solution.append(pc)
    return cart_solution

def GG(r,HK,L):
    ff=lambda phi,t: cos(tpi*r*(HK*sqrt(1-t*t)*sin(phi)-L*t))
    return dblquad(ff,-1.0,1.0,lambda x:0.0,lambda x:tpi)[0]/fpi

def study_GG():
    A = matrix_A('a.res')
    A=numpy.array([[1,0,0],[0,1,0],[0,0,1]])
    C,D=getCD(A)
    T = matrix_inv(A)
    print(C)
    print(D)
    print(T)
    T11,T22,T33 = T[0][0],T[1][1],T[2][2]
    T12,T13,T23 = T[0][1],T[0][2],T[1][2]

    h,k=0,0 
    sl = 0.6   # 0.2, 0.6

    aa=T33
    bb=2*(T13*h+T23*k)
    cc=T11*h*h+T22*k*k+2*T12*h*k-(2*sl)*(2*sl)
    delta=bb*bb-4*aa*cc  

    if delta<0:
        print('bad h,k...')
        return

    l=(-bb+sqrt(delta))/2/aa  
    print('h,k,l=',h,k,l)

    hkl=numpy.array([h,k,l])
    HKL=numpy.matmul(hkl,D)
    H,K,L=HKL
    print('H,K,L=',H,K,L)

    HK=sqrt(H*H+K*K)

    r = 1.0 

    g2=GG(r,HK,L)
    print(r,sl,(h,k,l),g2)
    print(GG(r,0,2*sl))
    print(gg(r,sl))

def gg(r,sl):
    x=12.56637061*r*sl 
    return numpy.sin(x)/x

def cell_volume(A):
    return sqrt(numpy.linalg.det(A))

def notnear(pt,pts,s,A): # shortest
    ss = s*s
    for p in pts:
        x = (pt[0]-p[0],pt[1]-p[1],pt[2]-p[2])
        if dis(x,A)<ss:
            return False 
    return True

def notnear3(pt,solution,atomj,A): # shortest
    #return True # allow clustering
    for i,p in enumerate(solution):
        atom = atomj[i]
        if atom in the_heavy:
            dd = 2.2 
        else:
            dd = 1.2
        #dd=5.0 # special for 1ab1 locating S-S
        x = (pt[0]-p[0],pt[1]-p[1],pt[2]-p[2])
        ss=dd*dd
        if dis(x,A)<ss:
            return False 
    return True 

def notnear3s(pt,solution,atomj,A,the_heavy): # shortest
    # return True # allow clustering
    for i,p in enumerate(solution):
        atom = atomj[i]
        if atom in the_heavy:
            dd = 2.2 
        else:
            dd = 1.2
        #dd=2.6
        x = (pt[0]-p[0],pt[1]-p[1],pt[2]-p[2])
        ss=dd*dd
        if dis(x,A)<ss:
            return False 
    return True 

def trianglebonding(p,solution,A):
    # return False  # allow triangle bonding
    if len(solution)<2:
        return False
    x,y,z = p
    n = len(solution)
    for i in range(n-1):
        xi,yi,zi = solution[i]
        ri = dis([x-xi,y-yi,z-zi],A)
        if ri>2.56: continue
        for j in range(i+1,n):
            xj,yj,zj = solution[j]
            rj = dis([x-xj,y-yj,z-zj],A)
            if rj>2.56: continue
            rij = dis([xi-xj,yi-yj,zi-zj],A)
            if rij>2.56: continue
            if ri<2.56 and rj<2.56 and rij<2.56: return True 
    return False

def localminmax(f,p,s):
    # function f(x,y,z),p=(x0,y0,z0),s=(sx,sy,sz)
    x0,y0,z0=p 
    sx,sy,sz=s 
    def ai(x,y,z):
        return numpy.array([x*x,y*y,z*z,x*y,y*z,x*z,x,y,z,1.0])
    a,b=[],[]
    a.append(ai(x0,y0,z0))
    b.append(f(x0,y0,z0))
    x,y,z=x0+sx,y0,z0 
    a.append(ai(x,y,z))
    b.append(f(x,y,z))
    x,y,z=x0-sx,y0,z0 
    a.append(ai(x,y,z))
    b.append(f(x,y,z))
    x,y,z=x0,y0+sy,z0 
    a.append(ai(x,y,z))
    b.append(f(x,y,z))
    x,y,z=x0,y0-sy,z0 
    a.append(ai(x,y,z))
    b.append(f(x,y,z))
    x,y,z=x0,y0,z0+sz 
    a.append(ai(x,y,z))
    b.append(f(x,y,z))
    x,y,z=x0,y0,z0-sz 
    a.append(ai(x,y,z))
    b.append(f(x,y,z))
    x,y,z=x0+sx,y0+sy,z0 
    a.append(ai(x,y,z))
    b.append(f(x,y,z))
    x,y,z=x0-sx,y0,z0-sz 
    a.append(ai(x,y,z))
    b.append(f(x,y,z))
    x,y,z=x0,y0-sy,z0+sz 
    a.append(ai(x,y,z))
    b.append(f(x,y,z))
    x=numpy.vstack(a)
    a=numpy.linalg.solve(x,b)
    b=numpy.array([[2*a[0],a[3],a[5]],
                [a[3],2*a[1],a[4]],
                [a[5],a[4],2*a[2]]])
    c=numpy.array([-a[6],-a[7],-a[8]])
    x=numpy.linalg.solve(b,c)
    return x 




def analyze_simple_molecule(molecule): # no parentheses
    # CH3OH --> {'C':1, 'H':4, 'O':1}
    atoms = {}
    n = len(molecule)
    i = 0
    while i < n:
        symbol = molecule[i]
        i += 1
        if i < n:
            c = molecule[i]
            if c.islower():
                symbol += c 
                i += 1
        num = ''
        while True:
            if i < n:
                c = molecule[i]
                if c.isdigit():
                    num += c 
                    i += 1
                    if i == n:
                        break
                else:
                    break
            else:
                break
        if num == '':
            num = 1
        else:
            num = int(num)
        atoms[symbol] = atoms.setdefault(symbol, 0) + num
    return atoms

def remove_parentheses(molecule):
    # (CH3O)2CO-->C2H6O2C1O1

    # replace {} and [] with ()
    new_molecule = ''
    for i in range(len(molecule)):
        c = molecule[i]
        if c in ['{','[']:
            new_molecule += '('
        elif c in ['}',']']:
            new_molecule += ')'
        else:
            new_molecule += c

    j = new_molecule.find(')')
    if j==-1:
        return molecule 
    i = new_molecule[:j].rfind('(')
    if j==len(new_molecule)-1:
        return new_molecule[:i]+new_molecule[i+1:j]

    num = ''
    k = j+1
    while k < len(new_molecule):
        c = new_molecule[k]
        if c.isdigit():
            num += c
            k += 1
        else:
            break
    if num=='':
        num = 1
    else:
        num = int(num)
    part1 = new_molecule[:i]
    part2 = new_molecule[i+1:j]
    part3 = new_molecule[k:]
    atoms = analyze_simple_molecule(part2)
    for a in atoms:
        atoms[a] *= num 
    part2 = ''
    for a in atoms:
        part2 += a + str(atoms[a])
    return part1 + part2 + part3

def to_atoms_dict(molecule):
    # (CH3O)2CO-->{'C':3, 'H':6, 'O':3}
    old_molecule = molecule
    while True:
        new_molecule = remove_parentheses(old_molecule)
        if new_molecule==old_molecule:
            break
        else:
            old_molecule = new_molecule
    return analyze_simple_molecule(new_molecule)

def to_molecular_formula(atoms):
    # {'C':1, 'H':4, 'O':1}-->CH3OH
    formula = ''
    for atom in atoms:
        formula += str(atom)
        if atoms[atom]>1:
            formula += str(atoms[atom])
    return formula 

def molecular_weight(molecule):
    atoms = to_atoms_dict(molecule)
    MW = 0
    for atom in atoms:
        n = atoms[atom]           # number of atoms
        M = elements[atom]['M']   # atomic weight
        MW += n * M 
    return MW 

def density(molecule,Z,A):
    # return density in g/cm^3
    MW = molecular_weight(molecule)
    cell_V = cell_volume(A)
    return Z*MW/0.6022/cell_V

def get_content(molecule,Z):
    atoms = to_atoms_dict(molecule)
    for k in atoms:
        atoms[k] *= Z 
    return atoms 



def kill_half_hkl(h,k,l,F2):
    hp,kp,lp,F2p = [],[],[],[]
    for i in range(len(h)):
        if l[i]>0:
            hp.append(h[i])
            kp.append(k[i])
            lp.append(l[i])
            F2p.append(F2[i])
        elif l[i]==0:
            if k[i]>0:
                hp.append(h[i])
                kp.append(k[i])
                lp.append(l[i])
                F2p.append(F2[i])
            elif k[i]==0:
                if h[i]>0:
                    hp.append(h[i])
                    kp.append(k[i])
                    lp.append(l[i])
                    F2p.append(F2[i])
    hp,kp,lp,F2p = numpy.array(hp),numpy.array(kp),numpy.array(lp),numpy.array(F2p)
    return (hp,kp,lp,F2p)

# electron scattering factors
sct1 = {
 'C': (numpy.array([1.029, -1.078, 1.379, 1.137]),
       numpy.array([30.382, 1.285, 1.285, 7.889]),
       0.037),
 'Cl': (numpy.array([1.012, 1.056, 2.241, 0.475]),
        numpy.array([36.239, 4.127, 13.058, 0.772]),
        0.074),
 'N': (numpy.array([0.357, 0.628, 0.216, 0.979]),
       numpy.array([36.076, 3.99, 0.852, 12.401]),
       0.033),
 'O': (numpy.array([0.300, 0.613, 0.232, 0.803]),
       numpy.array([28.323, 3.51, 0.788, 10.271]),
       0.035),
}


sct = {
'Ac': (numpy.array([35.6597 , 23.1032 , 12.5977 ,  4.08655]),
        numpy.array([  0.589092,   3.65155 ,  18.599   , 117.02    ]),
        13.5266),
 'Ag': (numpy.array([19.2808, 16.6885,  4.8045,  1.0463]),
        numpy.array([ 0.6446,  7.4726, 24.6605, 99.8156]),
        5.179),
'Al': (numpy.array([6.4202, 1.9002, 1.5936, 1.9646]),
        numpy.array([ 3.0387,  0.7426, 31.5472, 85.0886]),
        1.1151),
 'Am': (numpy.array([36.6706 , 24.0992 , 17.3415 ,  3.49331]),
        numpy.array([  0.483629,   3.20647 ,  14.3136  , 102.273   ]),
        13.3592),
 'Ar': (numpy.array([7.4845, 6.7723, 0.6539, 1.6442]),
        numpy.array([ 0.9072, 14.8407, 43.8983, 33.3929]),
        1.4445),
 'As': (numpy.array([16.6723,  6.0701,  3.4313,  4.2779]),
        numpy.array([ 2.6345,  0.2647, 12.9479, 47.7972]),
        2.531),
 'At': (numpy.array([35.3163 , 19.0211 ,  9.49887,  7.42518]),
        numpy.array([ 0.68587,  3.97458, 11.3824 , 45.4715 ]),
        13.7108),
 'Au': (numpy.array([16.8819, 18.5913, 25.5582,  5.86  ]),
        numpy.array([ 0.4611,  8.6216,  1.4826, 36.3956]),
        12.0658),
 'B': (numpy.array([2.0545, 1.3326, 1.0979, 0.7068]),
       numpy.array([23.2185,  1.021 , 60.3498,  0.1403]),
       -0.1932),
 'Ba': (numpy.array([20.3361, 19.297 , 10.888 ,  2.6959]),
        numpy.array([  3.216 ,   0.2756,  20.2073, 167.202 ]),
        2.7731),
 'Be': (numpy.array([1.5919, 1.1278, 0.5391, 0.7029]),
        numpy.array([ 43.6427,   1.8623, 103.483 ,   0.542 ]),
        0.0385),
 'Bi': (numpy.array([33.3689, 12.951 , 16.5877,  6.4692]),
        numpy.array([ 0.704 ,  2.9238,  8.7937, 48.0093]),
        13.5782),
 'Bk': (numpy.array([36.7881 , 24.7736 , 17.8919 ,  4.23284]),
        numpy.array([ 0.451018,  3.04619 , 12.8946  , 86.003   ]),
        13.2754),
 'Br': (numpy.array([17.1789,  5.2358,  5.6377,  3.9851]),
        numpy.array([ 2.1723, 16.5796,  0.2609, 41.4328]),
        2.9557),
 'C': (numpy.array([2.31  , 1.02  , 1.5886, 0.865 ]),
       numpy.array([20.8439, 10.2075,  0.5687, 51.6512]),
       0.2156),
 'Ca': (numpy.array([8.6266, 7.3873, 1.5899, 1.0211]),
        numpy.array([ 10.4421,   0.6599,  85.7484, 178.437 ]),
        1.3751),
 'Cd': (numpy.array([19.2214, 17.6444,  4.461 ,  1.6029]),
        numpy.array([ 0.5946,  6.9089, 24.7008, 87.4825]),
        5.0694),
 'Ce': (numpy.array([21.1671 , 19.7695 , 11.8513 ,  3.33049]),
        numpy.array([  2.81219 ,   0.226836,  17.6083  , 127.113   ]),
        1.86264),
 'Cf': (numpy.array([36.9185 , 25.1995 , 18.3317 ,  4.24391]),
        numpy.array([ 0.437533,  3.00775 , 12.4044  , 83.7881  ]),
        13.2674),
 'Cl': (numpy.array([11.4604,  7.1964,  6.2556,  1.6455]),
        numpy.array([1.04000e-02, 1.16620e+00, 1.85194e+01, 4.77784e+01]),
        -9.5574),
 'Cm': (numpy.array([36.6488 , 24.4096 , 17.399  ,  4.21665]),
        numpy.array([ 0.465154,  3.08997 , 13.4346  , 88.4834  ]),
        13.2887),
 'Co': (numpy.array([12.2841,  7.3409,  4.0034,  2.3488]),
        numpy.array([ 4.2791,  0.2784, 13.5359, 71.1692]),
        1.0118),
 'Cr': (numpy.array([10.6406,  7.3537,  3.324 ,  1.4922]),
        numpy.array([ 6.1038,  0.392 , 20.2626, 98.7399]),
        1.1832),
 'Cs': (numpy.array([20.3892, 19.1062, 10.662 ,  1.4953]),
        numpy.array([  3.569 ,   0.3107,  24.3879, 213.904 ]),
        3.3352),
 'Cu': (numpy.array([13.338 ,  7.1676,  5.6158,  1.6735]),
        numpy.array([ 3.5828,  0.247 , 11.3966, 64.8126]),
        1.191),
 'Dy': (numpy.array([26.507  , 17.6383 , 14.5596 ,  2.96577]),
        numpy.array([  2.1802  ,   0.202172,  12.1899  , 111.874   ]),
        4.29728),
 'Er': (numpy.array([27.6563 , 16.4285 , 14.9779 ,  2.98233]),
        numpy.array([  2.07356 ,   0.223545,  11.3604  , 105.703   ]),
        5.92046),
 'Eu': (numpy.array([24.6274, 19.0886, 13.7603,  2.9227]),
        numpy.array([  2.3879,   0.1942,  13.7546, 123.174 ]),
        2.5745),
 'F': (numpy.array([3.5392, 2.6412, 1.517 , 1.0243]),
       numpy.array([10.2825,  4.2944,  0.2615, 26.1476]),
       0.2776),
 'Fe': (numpy.array([11.7695,  7.3573,  3.5222,  2.3045]),
        numpy.array([ 4.7611,  0.3072, 15.3535, 76.8805]),
        1.0369),
 'Fr': (numpy.array([35.9299 , 23.0547 , 12.1439 ,  2.11253]),
        numpy.array([  0.646453,   4.17619 ,  23.1052  , 150.645   ]),
        13.7247),
 'Ga': (numpy.array([15.2354,  6.7006,  4.3591,  2.9623]),
        numpy.array([ 3.0669,  0.2412, 10.7805, 61.4135]),
        1.7189),
 'Gd': (numpy.array([25.0709 , 19.0798 , 13.8518 ,  3.54545]),
        numpy.array([  2.25341 ,   0.181951,  12.9331  , 101.398   ]),
        2.4196),
 'Ge': (numpy.array([16.0816,  6.3747,  3.7068,  3.683 ]),
        numpy.array([ 2.8509,  0.2516, 11.4468, 54.7625]),
        2.1313),
 'H': (numpy.array([0.489918, 0.262003, 0.196767, 0.049879]),
       numpy.array([20.6593 ,  7.74039, 49.5519 ,  2.20159]),
       0.001305),
 'He': (numpy.array([0.8734, 0.6309, 0.3112, 0.178 ]),
        numpy.array([ 9.1037,  3.3568, 22.9276,  0.9821]),
        0.0064),
 'Hf': (numpy.array([29.144  , 15.1726 , 14.7586 ,  4.30013]),
        numpy.array([ 1.83262 ,  9.5999  ,  0.275116, 72.029   ]),
        8.58154),
 'Hg': (numpy.array([20.6809, 19.0417, 21.6575,  5.9676]),
        numpy.array([ 0.545 ,  8.4484,  1.5729, 38.3246]),
        12.6089),
 'Ho': (numpy.array([26.9049 , 17.294  , 14.5583 ,  3.63837]),
        numpy.array([ 2.07051,  0.19794, 11.4407 , 92.6566 ]),
        4.56796),
 'I': (numpy.array([20.1472, 18.9949,  7.5138,  2.2735]),
       numpy.array([ 4.347 ,  0.3814, 27.766 , 66.8776]),
       4.0712),
 'In': (numpy.array([19.1624, 18.5596,  4.2948,  2.0396]),
        numpy.array([ 0.5476,  6.3776, 25.8499, 92.8029]),
        4.9391),
 'Ir': (numpy.array([27.3049 , 16.7296 , 15.6115 ,  5.83377]),
        numpy.array([ 1.59279 ,  8.86553 ,  0.417916, 45.0011  ]),
        11.4722),
 'K': (numpy.array([8.2186, 7.4398, 1.0519, 0.8659]),
       numpy.array([ 12.7949,   0.7748, 213.187 ,  41.6841]),
       1.4228),
 'Kr': (numpy.array([17.3555,  6.7286,  5.5493,  3.5375]),
        numpy.array([ 1.9384, 16.5623,  0.2261, 39.3972]),
        2.825),
 'La': (numpy.array([20.578  , 19.599  , 11.3727 ,  3.28719]),
        numpy.array([  2.94817 ,   0.244475,  18.7726  , 133.124   ]),
        2.14678),
 'Li': (numpy.array([1.1282, 0.7508, 0.6175, 0.4653]),
        numpy.array([  3.9546,   1.0524,  85.3905, 168.261 ]),
        0.0377),
 'Lu': (numpy.array([28.9476 , 15.2208 , 15.1    ,  3.71601]),
        numpy.array([ 1.90182 ,  9.98519 ,  0.261033, 84.3298  ]),
        7.97628),
 'Mg': (numpy.array([5.4204, 2.1735, 1.2269, 2.3073]),
        numpy.array([ 2.8275, 79.2611,  0.3808,  7.1937]),
        0.8584),
 'Mn': (numpy.array([11.2819,  7.3573,  3.0193,  2.2441]),
        numpy.array([ 5.3409,  0.3432, 17.8674, 83.7543]),
        1.0896),
 'Mo': (numpy.array([ 3.7025, 17.2356, 12.8876,  3.7429]),
        numpy.array([ 0.2772,  1.0958, 11.004 , 61.6584]),
        4.3875),
 'N': (numpy.array([12.2126,  3.1322,  2.0125,  1.1663]),
       numpy.array([5.70000e-03, 9.89330e+00, 2.89975e+01, 5.82600e-01]),
       -11.529),
 'Na': (numpy.array([4.7626, 3.1736, 1.2674, 1.1128]),
        numpy.array([  3.285 ,   8.8422,   0.3136, 129.424 ]),
        0.676),
 'Nb': (numpy.array([17.6142 , 12.0144 ,  4.04183,  3.53346]),
        numpy.array([ 1.18865 , 11.766   ,  0.204785, 69.7957  ]),
        3.75591),
 'Nd': (numpy.array([22.6845 , 19.6847 , 12.774  ,  2.85137]),
        numpy.array([  2.66248 ,   0.210628,  15.885   , 137.903   ]),
        1.98486),
 'Ne': (numpy.array([3.9553, 3.1125, 1.4546, 1.1251]),
        numpy.array([ 8.4042,  3.4262,  0.2306, 21.7184]),
        0.3515),
 'Ni': (numpy.array([12.8376,  7.292 ,  4.4438,  2.38  ]),
        numpy.array([ 3.8785,  0.2565, 12.1763, 66.3421]),
        1.0341),
 'numpy': (numpy.array([36.1874, 23.5964, 15.6402,  4.1855]),
        numpy.array([ 0.511929,  3.25396 , 15.3622  , 97.4908  ]),
        13.3573),
 'O': (numpy.array([3.0485, 2.2868, 1.5463, 0.867 ]),
       numpy.array([13.2771,  5.7011,  0.3239, 32.9089]),
       0.2508),
 'Os': (numpy.array([28.1894 , 16.155  , 14.9305 ,  5.67589]),
        numpy.array([ 1.62903 ,  8.97948 ,  0.382661, 48.1647  ]),
        11.0005),
 'P': (numpy.array([6.4345, 4.1791, 1.78  , 1.4908]),
       numpy.array([ 1.9067, 27.157 ,  0.526 , 68.1645]),
       1.1149),
 'Pa': (numpy.array([35.8847 , 23.2948 , 14.1891 ,  4.17287]),
        numpy.array([  0.547751,   3.41519 ,  16.9235  , 105.251   ]),
        13.4287),
 'Pb': (numpy.array([31.0617, 13.0637, 18.442 ,  5.9696]),
        numpy.array([ 0.6902,  2.3576,  8.618 , 47.2579]),
        13.4118),
 'Pd': (numpy.array([19.3319  , 15.5017  ,  5.29537 ,  0.605844]),
        numpy.array([ 0.698655,  7.98929 , 25.2052  , 76.8986  ]),
        5.26593),
 'Pm': (numpy.array([23.3405 , 19.6095 , 13.1235 ,  2.87516]),
        numpy.array([  2.5627  ,   0.202088,  15.1009  , 132.721   ]),
        2.02876),
 'Po': (numpy.array([34.6726 , 15.4733 , 13.1138 ,  7.02588]),
        numpy.array([ 0.700999,  3.55078 ,  9.55642 , 47.0045  ]),
        13.677),
 'Pr': (numpy.array([22.044  , 19.6697 , 12.3856 ,  2.82428]),
        numpy.array([  2.77393 ,   0.222087,  16.7669  , 143.644   ]),
        2.0583),
 'Pt': (numpy.array([27.0059, 17.7639, 15.7131,  5.7837]),
        numpy.array([ 1.51293 ,  8.81174 ,  0.424593, 38.6103  ]),
        11.6883),
 'Pu': (numpy.array([36.5254 , 23.8083 , 16.7707 ,  3.47947]),
        numpy.array([  0.499384,   3.26371 ,  14.9455  , 105.98    ]),
        13.3812),
 'Ra': (numpy.array([35.763  , 22.9064 , 12.4739 ,  3.21097]),
        numpy.array([  0.616341,   3.87135 ,  19.9887  , 142.325   ]),
        13.6211),
 'Rb': (numpy.array([17.1784,  9.6435,  5.1399,  1.5292]),
        numpy.array([  1.7888,  17.3151,   0.2748, 164.934 ]),
        3.4873),
 'Re': (numpy.array([28.7621 , 15.7189 , 14.5564 ,  5.44174]),
        numpy.array([ 1.67191,  9.09227,  0.3505 , 52.0861 ]),
        10.472),
 'Rh': (numpy.array([19.2957 , 14.3501 ,  4.73425,  1.28918]),
        numpy.array([ 0.751536,  8.21758 , 25.8749  , 98.6062  ]),
        5.328),
 'Rn': (numpy.array([35.5631, 21.2816,  8.0037,  7.4433]),
        numpy.array([ 0.6631,  4.0691, 14.0422, 44.2473]),
        13.6905),
 'Ru': (numpy.array([19.2674 , 12.9182 ,  4.86337,  1.56756]),
        numpy.array([ 0.80852,  8.43467, 24.7997 , 94.2928 ]),
        5.37874),
 'S': (numpy.array([6.9053, 5.2034, 1.4379, 1.5863]),
       numpy.array([ 1.4679, 22.2151,  0.2536, 56.172 ]),
       0.8669),
 'Sb': (numpy.array([19.6418, 19.0455,  5.0371,  2.6827]),
        numpy.array([ 5.3034,  0.4607, 27.9074, 75.2825]),
        4.5909),
 'Sc': (numpy.array([9.189 , 7.3679, 1.6409, 1.468 ]),
        numpy.array([  9.0213,   0.5729, 136.108 ,  51.3531]),
        1.3329),
 'Se': (numpy.array([17.0006,  5.8196,  3.9731,  4.3543]),
        numpy.array([ 2.4098,  0.2726, 15.2372, 43.8163]),
        2.8409),
 'Si': (numpy.array([6.292,3.035,1.989,1.541]),
        numpy.array([2.439,32.334,0.678,81.694]),
        1.141),
 'Sm': (numpy.array([24.0042 , 19.4258 , 13.4396 ,  2.89604]),
        numpy.array([  2.47274 ,   0.196451,  14.3996  , 128.007   ]),
        2.20963),
 'Sn': (numpy.array([19.1889, 19.1005,  4.4585,  2.4663]),
        numpy.array([ 5.8303,  0.5031, 26.8909, 83.9571]),
        4.7821),
 'Sr': (numpy.array([17.5663,  9.8184,  5.422 ,  2.6694]),
        numpy.array([  1.5564,  14.0988,   0.1664, 132.376 ]),
        2.5064),
 'Ta': (numpy.array([29.2024 , 15.2293 , 14.5135 ,  4.76492]),
        numpy.array([ 1.77333 ,  9.37046 ,  0.295977, 63.3644  ]),
        9.24354),
 'Tb': (numpy.array([25.8976 , 18.2185 , 14.3167 ,  2.95354]),
        numpy.array([  2.24256 ,   0.196143,  12.6648  , 115.362   ]),
        3.58324),
 'Tc': (numpy.array([19.1301 , 11.0948 ,  4.64901,  2.71263]),
        numpy.array([ 0.864132,  8.14487 , 21.5707  , 86.8472  ]),
        5.40428),
 'Te': (numpy.array([19.9644 , 19.0138 ,  6.14487,  2.5239 ]),
        numpy.array([ 4.81742 ,  0.420885, 28.5284  , 70.8403  ]),
        4.352),
 'Th': (numpy.array([35.5645 , 23.4219 , 12.7473 ,  4.80703]),
        numpy.array([ 0.563359,  3.46204 , 17.8309  , 99.1722  ]),
        13.4314),
 'Ti': (numpy.array([9.7595, 7.3558, 1.6991, 1.9021]),
        numpy.array([  7.8508,   0.5   ,  35.6338, 116.105 ]),
        1.2807),
 'Tl': (numpy.array([27.5446 , 19.1584 , 15.538  ,  5.52593]),
        numpy.array([ 0.65515,  8.70751,  1.96347, 45.8149 ]),
        13.1746),
 'Tm': (numpy.array([28.1819 , 15.8851 , 15.1542 ,  2.98706]),
        numpy.array([  2.02859 ,   0.238849,  10.9975  , 102.961   ]),
        6.75621),
 'U': (numpy.array([36.0228, 23.4128, 14.9491,  4.188 ]),
       numpy.array([  0.5293,   3.3253,  16.0927, 100.613 ]),
       13.3966),
 'V': (numpy.array([10.2971,  7.3511,  2.0703,  2.0571]),
       numpy.array([  6.8657,   0.4385,  26.8938, 102.478 ]),
       1.2199),
 'W': (numpy.array([29.0818 , 15.43   , 14.4327 ,  5.11982]),
       numpy.array([ 1.72029 ,  9.2259  ,  0.321703, 57.056   ]),
       9.8875),
 'Xe': (numpy.array([20.2933, 19.0298,  8.9767,  1.99  ]),
        numpy.array([ 3.9282,  0.344 , 26.4659, 64.2658]),
        3.7118),
 'Y': (numpy.array([17.776  , 10.2946 ,  5.72629,  3.26588]),
       numpy.array([  1.4029  ,  12.8006  ,   0.125599, 104.354   ]),
       1.91213),
 'Yb': (numpy.array([28.6641 , 15.4345 , 15.3087 ,  2.98963]),
        numpy.array([  1.9889  ,   0.257119,  10.6647  , 100.417   ]),
        7.56672),
 'Zn': (numpy.array([14.0743,  7.0318,  5.1652,  2.41  ]),
        numpy.array([ 3.2655,  0.2333, 10.3163, 58.7097]),
        1.3041),
 'Zr': (numpy.array([17.8765 , 10.948  ,  5.41732,  3.65721]),
        numpy.array([ 1.27618 , 11.916   ,  0.117622, 87.6627  ]),
        2.06929)}



def output_sct():
    with open('sct.txt','w') as f:
        for atom in sct:
            sa,sb,sc=sct[atom]
            print(atom,sa[0],sa[1],sa[2],sa[3],sb[0],sb[1],sb[2],sb[3],sc,file=f)


def fsc(sl,atom): #U=0
    #return elements[atom]['Z']
    # sl = sin(th)/lambda
    sa,sb,sc = sct[atom]
    return (sa*exp(-sb*sl*sl)).sum()+sc

def fsc_simple(sl,atom): #U=0
    tail={'C':0.2,'S':0.88,'N':-1.0,'O':0.253,'I':4.06,'Mo':4.54,'W':10.12,}
    return (elements[atom]['Z']-tail[atom])*exp(-sl*sl)+tail[atom]
    # sl = sin(th)/lambda
    sa,sb,sc = sct[atom]
    return (sa*exp(-sb*sl*sl)).sum()+sc


def fsc2(sl,atom,U=0.05):
    # sl = sin(th)/lambda
    B = 8*pi*pi*U
    sa,sb,sc = sct[atom]
    return ((sa*exp(-sb*sl*sl)).sum()+sc)*exp(-B*sl*sl)


def get_sl(h,k,l,A):
    T = matrix_inv(A)
    T11,T22,T33 = T[0][0],T[1][1],T[2][2]
    T12,T13,T23 = T[0][1],T[0][2],T[1][2]

    # sl = sin(th)/lambda
    sl = sqrt(T11*h*h+T22*k*k+T33*l*l+2*T12*h*k+2*T13*h*l+2*T23*l*k)/2 
    return sl 

def output_matrix_T(A):
    a,b,c=abc(A)
    A11,A22,A33 = A[0][0],A[1][1],A[2][2]
    A12,A13,A23 = A[0][1],A[0][2],A[1][2]
    T = matrix_inv(A)
    T11,T22,T33 = T[0][0],T[1][1],T[2][2]
    T12,T13,T23 = T[0][1],T[0][2],T[1][2]
    with open('matrix_T.txt','w') as f:
        print(T11,T22,T33,file=f)
        print(T12,T13,T23,file=f)
        print(A11,A22,A33,file=f)
        print(A12,A13,A23,file=f)
        print(a,b,c,file=f)


def scaling_factor(content,sl,h,k,l,F2):
    # determine scaling factor
    total = 0
    for atom in content:
        fj = numpy.array([fsc(s,atom) for s in sl])
        F2c_total = (fj**2).sum()
        total += F2c_total*content[atom]
    total_obs = F2.sum()
    scale = total/total_obs
    return scale


def atomj_solution(atom_list):
    atomj,atom_labels,solution = [],[],[]
    for atom,label,x,y,z in atom_list:
        x,y,z = float(x),float(y),float(z)
        atomj.append(atom)
        atom_labels.append(label)
        solution.append((x,y,z))
    return (atomj,atom_labels,solution)


def to_atom_list(atomj,atom_labels,solution):
    atom_list = []
    for i,atom in enumerate(atomj):
        x,y,z = solution[i]
        atom_list.append((atom,atom_labels[i],x,y,z))
    return atom_list 



def save_solution(heavy_atom,heavy_label,nheavy,solution,light_label='1',
    light_atom='C',do_copy=False):
    if not do_copy: return # not to save solution, but copy to clipboard
    num = 0
    text = ''
    #with open('solution.txt','w') as f:
    if True:
        for x,y,z in solution:
            num += 1
            if num<nheavy+1:
                q = st3(heavy_atom[num-1]+str(num))+heavy_label[num-1]
            else:
                q = st3(light_atom+str(num))+light_label
            xt = st2(round(x+0,4))
            yt = st2(round(y+0,4))
            zt = st2(round(z+0,4))
            st = '  11.00 0.05  '
            line = q+xt+yt+zt+st 
            #print(line)
            #f.write(line+'\n')
            text += line+'\n'
    if do_copy:cp(text[:-1])
    #print('solution peaks are saved')

def r1_func(x,y,z,h,k,l,f2,fcorrection,Ah1,Bh1,Fo):
    angle = 6.283185306*(h[:,numpy.newaxis]*x[numpy.newaxis,:]+k[:,numpy.newaxis]*y[numpy.newaxis,:]+l[:,numpy.newaxis]*z[numpy.newaxis,:])
    Ahj = f2[:,numpy.newaxis] * numpy.sin(angle) 
    Bhj = f2[:,numpy.newaxis] * numpy.cos(angle) 
    Ah =Ah1[:,numpy.newaxis] + Ahj
    Bh =Bh1[:,numpy.newaxis] + Bhj 
    F2c = Ah**2+Bh**2
    Dh = numpy.sqrt(F2c+fcorrection[:,numpy.newaxis])-Fo[:,numpy.newaxis]
    return numpy.sum(abs(Dh),axis=0)

def r1_scalor(x,y,z,h,k,l,f2,fcorrection,Ah1,Bh1,Fo):
    angle = tpi*(h*x+k*y+l*z)
    Ahj = f2 * sin(angle) 
    Bhj = f2 * cos(angle) 
    Ah =Ah1 + Ahj
    Bh =Bh1 + Bhj 
    F2c = Ah**2+Bh**2
    Dh = sqrt(F2c+fcorrection)-Fo
    return numpy.sum(abs(Dh))







def save_peaks(h,k,l,Fo,A,molecule,Z,ntotal,atom_list,
    starttime=None,runs='sR1_method',more_info=None):
    # save peaks for running single atom global min to build model from scratch
    # ntoal = expand to
    # atom_list = already determined part
    # molecule and Z: expected atom list 

    if starttime is None: starttime = time.time()
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('Starting global min with dips...',tt)

    F2=Fo**2

    a,b,c = abc(A)

    sl = get_sl(h,k,l,A)

    # the part of molecule already finished:
    atomj,atom_labels,solution=atomj_solution(atom_list)

    # the part not finished yet:
    content=get_content(molecule,Z)
    Natoms=0  
    for atom in content:
        Natoms+=content[atom]
    for atom in atomj:
        content[atom]-=1

    do_special=False
    if more_info is not None: # extend single atom
        try:
            next_atom,next_label,jj,d0,dd=more_info #'C','1'
            content[next_atom]-=1
            atoms,labels=atoms_labels_from_content(content)
            atoms,labels=[next_atom]+atoms,[next_label]+labels 
        except:
            try:
                next_atom,next_label,jj,nn=more_info #'C','1'
                content[next_atom]-=1
                atoms,labels=atoms_labels_from_content(content)
                atoms,labels=[next_atom]+atoms,[next_label]+labels 
            except:
                try:
                    jj,NN=more_info
                    do_special=True  
                    L=max(a,b,c)
                    na,nb,nc=int(a/0.4),int(b/0.4),int(c/0.4)
                    Na,Nb,Nc=int((2*NN+1)*a/L),int((2*NN+1)*b/L),int((2*NN+1)*c/L)
                    NC=int(Natoms*Na*Nb*Nc/na/nb/nc)
                    content['C']-=NC  
                    atoms,labels=atoms_labels_from_content(content)
                    atoms,labels=['C']*NC+atoms,['1']*NC+labels 
                except:
                    print('incorrect more_info')
                    return
    else:
        atoms,labels=atoms_labels_from_content(content)

    # the whole molecule:
    heavy_atom=atomj+atoms 
    heavy_label=atom_labels+labels  
    Nheavy=len(heavy_atom)

    # all atomic scattering factors
    f2a = {}
    for atom in heavy_atom:
        if atom not in f2a:
            f2a[atom] = numpy.array([fsc(s,atom) for s in sl]) 
    f2S = numpy.array([f2a[heavy_atom[i]]  for i in range(Nheavy)])

    # the heaviest unfinished atomic factor
    jj=len(atomj)
    f2 = f2S[jj]

    # calculate correction
    for i in range(jj,len(f2S)):
        if i == jj:
            fcorrection = f2S[i]**2
        else:
            fcorrection += f2S[i]**2

    xj = numpy.array([s[0] for s in solution])
    yj = numpy.array([s[1] for s in solution])
    zj = numpy.array([s[2] for s in solution])

    fj_tmp =f2S[:jj]
    fj=fj_tmp.T

    Ahj = (fj * sin(tpi*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
        +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
    Bhj = (fj * cos(tpi*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
        +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
    Ah1 =numpy.sum(Ahj ,axis=-1)
    Bh1 =numpy.sum(Bhj ,axis=-1)

    iheavy=jj
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print("jj = ",jj,tt)
    with open('history.txt','a') as f:
        print("jj = ",jj,tt,file=f)

    s_precision =  0.4
    s = s_precision 

    peaks_raw=filter(h,k,l,F2,A,heavy_atom,atom_list,starttime,more_info=more_info)

    peaks=[]
    for x,y,z,ff in peaks_raw:
        if notnear3((x,y,z),solution,atomj,A):
            peaks.append((x,y,z,ff))

    x,y,z,ff_min=peaks[0]
    x,y,z,ff_max=peaks[-1]
    peak_list=[]
    for i in range(len(xj)):
        x,y,z=xj[i],yj[i],zj[i]
        peak_list.append((heavy_atom[i],heavy_label[i],x,y,z,0.05))
    for x,y,z,ff in peaks:
        hh=0.2+(0.005-0.2)*(ff-ff_min)/(ff_max-ff_min)
        hh=round(hh,4)
        peak_list.append(('C','1',x,y,z,hh))

    with open('peak_list.txt','w') as f:
        for atom,label,x,y,z,hh in peak_list:
            print(atom,label,x,y,z,hh,file=f)


    if 0:
        with open('peaks.txt','w') as f:
            for i in range(len(xj)):
                print(xj[i],yj[i],zj[i],0.1)
            for x,y,z,ff in peaks:
                print(x,y,z,ff,file=f)
        with open('history.txt','a') as f:
            print('\n\n\n\npeaks found:',file=f)
            if 0:
                for i in range(len(xj)):
                    print(xj[i],yj[i],zj[i],0.1,file=f)
            for x,y,z,ff in peaks:
                print(x,y,z,ff,file=f)
            print('\n\n\n',file=f)
    print('peaks saved')
    print('all done')
    return peak_list








def get_min(X,Y,Z,h,k,l,f2,fcorrection,Ah1,Bh1,Fo,Fosum):
    ncuts=4
    Ntot=len(X)
    Ncut= int(Ntot/ncuts)+1
    N1,N2,r1min,p_found=0,Ncut,1e100,None  
    while N1<Ntot:
        Xp,Yp,Zp=X[N1:N2],Y[N1:N2],Z[N1:N2]
        idx=numpy.argmin(r1_func(Xp,Yp,Zp,h,k,l,f2,fcorrection,Ah1,Bh1,Fo))
        x,y,z = Xp[idx],Yp[idx],Zp[idx]
        angle = 6.283185306*(h*x+k*y+l*z)
        Ahj = f2 * numpy.sin(angle) 
        Bhj = f2 * numpy.cos(angle) 
        Ah2 = Ah1 + Ahj 
        Bh2 = Bh1 + Bhj 
        F2c = Ah2**2+Bh2**2
        r1 = (abs(numpy.sqrt(F2c+fcorrection)-Fo)).sum()/Fosum
        N1,N2=N1+Ncut,N2+Ncut
        if r1<r1min:
            r1min=r1
            p_found=x,y,z  
    return (r1min,p_found)



def globalmin_using_dips(h,k,l,Fo,A,molecule,Z,ntotal,atom_list,
    starttime=None,runs='sR1_method',more_info=None):
    # run single atom global min to build model from scratch
    # ntoal = expand to
    # atom_list = already determined part
    # molecule and Z: expected atom list 

    runs='sR1'

    Z_atoms=Z 

    if starttime is None: starttime = time.time()
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('Starting global min with dips...',tt)

    F2=Fo**2

    a,b,c = abc(A)

    #sl = get_sl(h,k,l,A)

    # the part of molecule already finished:
    atomj,atom_labels,solution=atomj_solution(atom_list)

    # the part not finished yet:
    content=get_content(molecule,Z)
    Natoms=0  
    for atom in content:
        Natoms+=content[atom]
    for atom in atomj:
        content[atom]-=1

    do_special=False
    bond_length_guided=False
    if more_info is not None: # extend single atom
        try:
            next_atom,jj,d0,dd=more_info #'C','1'
            next_label=label_dic.setdefault(next_atom,'1')
            jj-=1
            content[next_atom]-=1
            atoms,labels=atoms_labels_from_content(content)
            atoms,labels=[next_atom]+atoms,[next_label]+labels 
            bond_length_guided=True  
        except:
            try:
                next_atom,jj,nn=more_info #'C','1'
                next_label=label_dic.setdefault(next_atom,'1')
                jj-=1
                content[next_atom]-=1
                atoms,labels=atoms_labels_from_content(content)
                atoms,labels=[next_atom]+atoms,[next_label]+labels 
            except:
                try:
                    jj,NN=more_info
                    jj-=1
                    do_special=True  
                    L=max(a,b,c)
                    na,nb,nc=int(a/0.4),int(b/0.4),int(c/0.4)
                    Na,Nb,Nc=int((2*NN+1)*a/L),int((2*NN+1)*b/L),int((2*NN+1)*c/L)
                    NC=int(Natoms*Na*Nb*Nc/na/nb/nc)
                    content['C']-=NC  
                    atoms,labels=atoms_labels_from_content(content)
                    atoms,labels=['C']*NC+atoms,['1']*NC+labels 
                except:
                    print('incorrect more_info')
                    return
    else:
        atoms,labels=atoms_labels_from_content(content)

    # the whole molecule:
    if 0:
        heavy_atom=atomj+atoms[2:]+atoms[:2] 
    else:
        heavy_atom=atomj+atoms
    heavy_label=atom_labels+labels  
    Nheavy=len(heavy_atom)

    # all atomic scattering factors
    # f2a = {}
    # for atom in heavy_atom:
    #     if atom not in f2a:
    #         f2a[atom] = numpy.array([fsc(s,atom) for s in sl]) 
    f2S = numpy.array([f2a[heavy_atom[i]]  for i in range(Nheavy)])

    # the heaviest unfinished atomic factor
    jj=len(atomj)
    f2 = f2S[jj]

    # calculate correction
    fcorrection=0*f2**2  
    for i in range(jj,len(f2S)):
        if i == jj:
            fcorrection = f2S[i]**2
        else:
            fcorrection += f2S[i]**2

    xj = numpy.array([s[0] for s in solution])
    yj = numpy.array([s[1] for s in solution])
    zj = numpy.array([s[2] for s in solution])

    fj_tmp =f2S[:jj]
    fj=fj_tmp.T

    chj=(tpi*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
        +l[:,numpy.newaxis]*zj[numpy.newaxis,:]))
    Ahj = (fj * sin(chj) )
    Bhj = (fj * cos(chj) )
    Ah1 =numpy.sum(Ahj ,axis=-1)
    Bh1 =numpy.sum(Bhj ,axis=-1)

    iheavy=jj
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print("jj = ",jj,tt)
    with open('history.txt','a') as f:
        print("jj = ",jj,tt,file=f)

    s_precision =  0.4
    s = s_precision 

    peaks=filter(h,k,l,F2,A,heavy_atom,atom_list,starttime,more_info=more_info)
    if 0:
        with open('peaks.txt','w') as f:
            for i in range(len(xj)):
                print(xj[i],yj[i],zj[i],0.1)
            for x,y,z,ff in peaks:
                print(x,y,z,ff,file=f)
        with open('history.txt','a') as f:
            print('\n\n\n\npeaks found:',file=f)
            if 0:
                for i in range(len(xj)):
                    print(xj[i],yj[i],zj[i],0.1,file=f)
            for x,y,z,ff in peaks:
                print(x,y,z,ff,file=f)
            print('\n\n\n',file=f)
        print('peaks saved')
        sys.exit()


    X,Y,Z=[],[],[]
    for x,y,z,ff in peaks:
        if (not notnear3((x,y,z),solution,atomj,A)) or trianglebonding((x,y,z),solution,A):
            pass 
        else:
            X.append(x),Y.append(y),Z.append(z)
    X,Y,Z=numpy.array(X),numpy.array(Y),numpy.array(Z)

    xp,yp,zp=xpypzp(A)

    Fosum=Fo.sum()
 
    nextend = iheavy-1
    r1=(abs(sqrt(Ah1**2+Bh1**2+fcorrection)-Fo)).sum()/Fosum 
    if do_special:
        ntotal=len(solution)+NC  

    previoustime=time.time()

    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('expand to '+str(ntotal)+' atoms', time.time()-starttime,tt)

    r1_best=r1  
    solution_keep=[0.0]*len(solution)
    previous_r1=r1 
    print('r1 start = ',r1)
    with open('history.txt','a') as f:
        print('r1 start = ',r1,file=f)
    bad_count=0
    while len(solution) < ntotal:
        nextend += 1
        f2 = f2S[iheavy]

        fcorrection -= f2**2

        Ntot=len(X)
        Ncut=int(Ntot/ncpus)+1
        N1,N2,jobs=0,Ncut,[] 
        while N1<Ntot:
            jobs.append(job_server.submit(get_min,(X[N1:N2],Y[N1:N2],Z[N1:N2],
                h,k,l,f2,fcorrection,Ah1,Bh1,Fo,Fosum,),(r1_func,),('numpy',)))
            N1,N2=N1+Ncut,N2+Ncut 

        r1min,p_found=1e100,None  
        for job in jobs:
            r1,p=job()
            if r1<r1min:
                r1min=r1 
                p_found=p 

        precision = s_precision 
        while precision > 0.2: # was 0.001
            # improve precision
            precision /= 2 
            lim = precision
            x0,y0,z0 = p_found
            x = numpy.array([x0-lim/a,x0,x0+lim/a])  
            y = numpy.array([y0-lim/b,y0,y0+lim/b])  
            z = numpy.array([z0-lim/c,z0,z0+lim/c])  
            Xp,Yp,Zp = numpy.meshgrid(x,y,z)
            Xp,Yp,Zp=Xp.flatten(),Yp.flatten(),Zp.flatten()
            Mp=numpy.zeros_like(Xp)
            for idx,x in numpy.ndenumerate(Xp):
                y,z=Yp[idx],Zp[idx]
                if (not notnear3((x,y,z),solution,atomj,A)) or trianglebonding((x,y,z),solution,A):
                    Mp[idx]=1e100
            idx=numpy.argmin(r1_func(Xp,Yp,Zp,h,k,l,f2,fcorrection,Ah1,Bh1,Fo)/Fosum+Mp)
            p_found=Xp[idx],Yp[idx],Zp[idx]

        (x,y,z)=p_found
        p_found=(put_in_cell(x),put_in_cell(y),put_in_cell(z))
        solution.append(p_found)
        atomj.append(heavy_atom[iheavy]) 
        atom_labels.append(heavy_label[iheavy])

        asolution = do_arrange(solution,A)
        csolution=to_cartesian_solution(asolution,xp,yp,zp,A)

        x,y,z = solution[-1]
        angle = tpi*(h*x+k*y+l*z)
        Ahj = f2 * sin(angle) 
        Bhj = f2 * cos(angle) 
        Ah1 += Ahj 
        Bh1 += Bhj 

        F2c = Ah1**2+Bh1**2
        r1 = (abs(sqrt(F2c+fcorrection)-Fo)).sum()/Fosum
        #save_solution(heavy_atom,heavy_label,Nheavy,solution,light_label,light_atom)
        timenow = time.time()
        timeinterval = timenow-previoustime
        totaltime = timenow-starttime
        previoustime = timenow
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        print(runs,iheavy+1,round(r1,5),previous_r1-r1,int(10*timeinterval)/10,tt)
        with open('history.txt','a') as f:
            print(runs,iheavy+1,round(r1,5),previous_r1-r1,int(10*timeinterval)/10,tt,file=f)
        solution_keep.append(previous_r1-r1)
        if r1<r1_best: r1_best=r1 
        if r1<previous_r1:
            bad_count=0
        else:
            bad_count+=1  
            if bad_count>2000000: break
        previous_r1=r1 
        iheavy += 1

        if len(solution)==ntotal: break

        Xp,Yp,Zp=[],[],[]
        for idx,x in numpy.ndenumerate(X):
            y,z=Y[idx],Z[idx]
            if (not notnear3((x,y,z),solution,atomj,A)) or trianglebonding((x,y,z),solution,A):
                pass 
            else:
                Xp.append(x)
                Yp.append(y)
                Zp.append(z)
        X,Y,Z=numpy.array(Xp),numpy.array(Yp),numpy.array(Zp)  

    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('finished, solution saved'+tt)
    # with open('history.txt','a') as f:
    #     print('\n\n\nfinal result of ',runs,'time used: ',int(time.time()-starttime),tt,file=f)

    solution = do_arrange(solution,A)

    atom_list = to_atom_list(atomj,atom_labels,solution)
    content=get_content(molecule,Z_atoms)
    if not bond_length_guided: atom_list=relax_model(atom_list,f2a,h,k,l,Fo,Fosum,A,content,starttime)
    r1_best=get_sR1(atom_list,h,k,l,f2a,sl,Fo,Fosum,content)
    if not bond_length_guided: atom_list.sort(key=lambda xx:-elements[xx[0]]['Z'])
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('Finished global min with dips!',tt,r1_best)
    return (atom_list,r1_best,solution_keep)



def filter(h,k,l,F2,A,heavy_atom,atom_list,starttime,more_info=None):
    # locate all sR1 holes 
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('Searching peaks...',tt)

    a,b,c = abc(A)

    # to use all grid points:
    if False:
        s=0.4
        na,nb,nc = int(a/s),int(b/s),int(c/s)
        x = numpy.array([i/na for i in range(0,na)])
        y = numpy.array([i/nb for i in range(0,nb)])
        z = numpy.array([i/nc for i in range(0,nc)])
        X,Y,Z = numpy.meshgrid(x,y,z)
        peaks=[]
        X=X.flatten()
        Y=Y.flatten()
        Z=Z.flatten()
        for i in range(len(X)):
            x,y,z=X[i],Y[i],Z[i]
            peaks.append((x,y,z,1.0))
        return peaks 

    do_special=False
    # to use nearby grid points:
    if more_info is not None: # extend single atom
        try:
            next_atom,j,d0,dd=more_info #'C','1'
            next_label=label_dic.setdefault(next_atom,'1')
            j-=1
            p0=[atom_list[j][2],atom_list[j][3],atom_list[j][4]]
            x0,y0,z0=p0
            dmin,dmax=(d0-dd)*(d0-dd),(d0+dd)*(d0+dd)
            s=0.4
            na,nb,nc = int(a/s),int(b/s),int(c/s)
            nn=int((d0+dd)/s)+1  
            x=numpy.array([i/na for i in range(-nn,nn+1)])
            y=numpy.array([i/nb for i in range(-nn,nn+1)])
            z=numpy.array([i/nc for i in range(-nn,nn+1)])
            X,Y,Z=numpy.meshgrid(x,y,z)
            X,Y,Z=X.flatten(),Y.flatten(),Z.flatten()
            peaks=[]
            for i in range(len(X)):
                x,y,z=X[i],Y[i],Z[i]
                d=dis_exact([x,y,z],A)
                if dmin<d<dmax:
                    peaks.append((x0+x,y0+y,z0+z,1.0))
            return peaks 
        except:
            try:
                next_atom,j,nn=more_info #'C','1'
                next_label=label_dic.setdefault(next_atom,'1')
                j-=1
                p0=[atom_list[j][2],atom_list[j][3],atom_list[j][4]]
                x0,y0,z0=p0
                s=0.4
                na,nb,nc = int(a/s),int(b/s),int(c/s)
                x=numpy.array([i/na for i in range(-nn,nn+1)])
                y=numpy.array([i/nb for i in range(-nn,nn+1)])
                z=numpy.array([i/nc for i in range(-nn,nn+1)])
                X,Y,Z=numpy.meshgrid(x,y,z)
                X,Y,Z=X.flatten(),Y.flatten(),Z.flatten()
                peaks=[]
                for i in range(len(X)):
                    x,y,z=X[i],Y[i],Z[i]
                    peaks.append((x0+x,y0+y,z0+z,1.0))
                return peaks 
            except:
                try:
                    jj,NN=more_info
                    jj-=1
                    do_special=True  
                except:
                    print('incorrect more_info')
                    return



    Fo=sqrt(F2)
    Fosum=Fo.sum()

    Ntotal=len(heavy_atom)

    # sl = get_sl(h,k,l,A)
    # f2a = {}
    # for atom in heavy_atom:
    #     if atom not in f2a:
    #         f2a[atom] = numpy.array([fsc(s,atom) for s in sl]) 
    f2S = numpy.array([f2a[heavy_atom[i]]  for i in range(Ntotal)])


    # known part
    atomj,atom_labels,solution=atomj_solution(atom_list)

    startfrom = len(solution)
    f2 = f2S[startfrom]

    # calculate correction
    fcorrection=0*f2**2  # need this in case there are no additional missing atoms
    for i in range(startfrom+1,Ntotal):
        if i == startfrom+1:
            fcorrection = f2S[i]**2
        else:
            fcorrection += f2S[i]**2


    xj = numpy.array([s[0] for s in solution])
    yj = numpy.array([s[1] for s in solution])
    zj = numpy.array([s[2] for s in solution])

    fj_tmp =f2S[:startfrom]
    fj=fj_tmp.T

    chj = (tpi*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
        +l[:,numpy.newaxis]*zj[numpy.newaxis,:]))
    Ahj = (fj * sin(chj) )
    Bhj = (fj * cos(chj) )
    Ah1 =numpy.sum(Ahj ,axis=-1)
    Bh1 =numpy.sum(Bhj ,axis=-1)

    def rou12(x,y,z,h,k,l,f2,Ah1,Bh1,Fosum,Fo,fcorrection):
        angle = 6.283185306*(h*x+k*y+l*z)
        Ahj = f2 * numpy.sin(angle) 
        Bhj = f2 * numpy.cos(angle) 
        Ah =Ah1 + Ahj
        Bh =Bh1 + Bhj 
        Fc = numpy.sqrt(Ah**2+Bh**2+fcorrection)
        r1 = (abs(Fc-Fo)).sum()/Fosum
        return -r1


    def rou22(x,y,z,h,k,l,f2,Ah1,Bh1,Fosum,Fo,fcorrection):
        angle = 6.283185306*(h*x+k*y+l*z)
        Ahj = f2 * numpy.sin(angle) 
        Bhj = f2 * numpy.cos(angle) 
        Ah =Ah1 + Ahj
        Bh =Bh1 + Bhj 
        Fc = numpy.sqrt(Ah**2+Bh**2+fcorrection)
        r1 = abs(Fc-Fo)/Fosum
        return -r1


    runs="refining"

    global SIN  
    #SIN=None  
    s=0.4
    na,nb,Nc = int(a/s),int(b/s),int(c/s)
    nc=Nc 
    if SIN is None:
        if do_special:
            x0,y0,z0=solution[jj]
            L=max(a,b,c)
            Na,Nb,Nc=int(NN*a/L),int(NN*b/L),int(NN*c/L)
            X = numpy.array([x0+i/na for i in range(-Na,Na+1)])
            Y = numpy.array([y0+i/nb for i in range(-Nb,Nb+1)])
            Z0 = numpy.array([z0+i/Nc for i in range(-Nc-1,Nc+2)])
            nc=2*Nc+1  
        else:
            X = numpy.array([i/na for i in range(na)])
            Y = numpy.array([i/nb for i in range(nb)])
            Z0 = numpy.array([i/Nc for i in range(-1,Nc+1)])
            nc=Nc 
            if 0:
                X = numpy.array([i/na for i in range(int(na/2))])
            if 0:
                Y = numpy.array([i/nb for i in range(int(nb/2))])
            if 0:
                Z0 = numpy.array([i/Nc for i in range(-1,int(Nc/2)+1)])
                nc=int(Nc/2) 


    def get_peaks1(XX,Y,Z0,h,k,l,f2,Ah1,Bh1,Fosum,Fo,fcorrection,na,nb,nc):
        peaks = []
        for x in XX:
            for y in Y:
                R0=numpy.sum(rou22((x*numpy.ones_like(Z0))[:,numpy.newaxis],
                    (y*numpy.ones_like(Z0))[:,numpy.newaxis],Z0[:,numpy.newaxis],
                    h,k,l,f2,Ah1,Bh1,Fosum,Fo,fcorrection),axis=-1)
                R1=R0[:nc]
                R=R0[1:nc+1]
                Z=Z0[1:nc+1]
                R2=R0[2:nc+2]
                Rs1=R[(R1<R) * (R>R2)]
                Zs1=Z[(R1<R) * (R>R2)]
                Rs1x1=numpy.sum(rou22(((x-1/na)*numpy.ones_like(Zs1))[:,numpy.newaxis],
                    (y*numpy.ones_like(Zs1))[:,numpy.newaxis],Zs1[:,numpy.newaxis], 
                    h,k,l,f2,Ah1,Bh1,Fosum,Fo,fcorrection),axis=-1)
                Rs1x2=numpy.sum(rou22(((x+1/na)*numpy.ones_like(Zs1))[:,numpy.newaxis],
                    (y*numpy.ones_like(Zs1))[:,numpy.newaxis],Zs1[:,numpy.newaxis], 
                    h,k,l,f2,Ah1,Bh1,Fosum,Fo,fcorrection),axis=-1)
                Rs2=Rs1[(Rs1x1<Rs1) * (Rs1>Rs1x2)]
                Zs2=Zs1[(Rs1x1<Rs1) * (Rs1>Rs1x2)]
                Rs2y1=numpy.sum(rou22((x*numpy.ones_like(Zs2))[:,numpy.newaxis],
                    ((y-1/nb)*numpy.ones_like(Zs2))[:,numpy.newaxis],Zs2[:,numpy.newaxis], 
                    h,k,l,f2,Ah1,Bh1,Fosum,Fo,fcorrection),axis=-1)
                Rs2y2=numpy.sum(rou22((x*numpy.ones_like(Zs2))[:,numpy.newaxis],
                    ((y+1/nb)*numpy.ones_like(Zs2))[:,numpy.newaxis],Zs2[:,numpy.newaxis],
                    h,k,l,f2,Ah1,Bh1,Fosum,Fo,fcorrection),axis=-1)
                Rs3=Rs2[(Rs2y1<Rs2) * (Rs2>Rs2y2)]
                Zs3=Zs2[(Rs2y1<Rs2) * (Rs2>Rs2y2)]
                for i in range(len(Zs3)):
                    z,ff=Zs3[i],Rs3[i]
                    peaks.append((x,y,z,ff)) 
        return peaks 

    if SIN is None:
        jobs=[]
        nX=len(X)
        dn=int(nX/ncpus)+1 
        n1,n2=-dn,0 
        for i in range(ncpus):
            n1,n2=n1+dn,n2+dn 
            jobs.append(job_server.submit(get_peaks1,(X[n1:n2],Y,Z0,h,k,l,f2,Ah1,Bh1,
                Fosum,Fo,fcorrection,na,nb,nc),
                (rou22,),('numpy','math',)))
        peaks=[]
        for job in jobs:
            peaks+=job()
    else:
        Ahj = f2[numpy.newaxis,numpy.newaxis,numpy.newaxis,:] * SIN
        Bhj = f2[numpy.newaxis,numpy.newaxis,numpy.newaxis,:] * COS 
        Ah =Ah1[numpy.newaxis,numpy.newaxis,numpy.newaxis,:] + Ahj
        Bh =Bh1[numpy.newaxis,numpy.newaxis,numpy.newaxis,:] + Bhj 
        Fc = numpy.sqrt(Ah**2+Bh**2+fcorrection[numpy.newaxis,numpy.newaxis,numpy.newaxis,:])
        r1 = -numpy.sum(abs(Fc-Fo[numpy.newaxis,numpy.newaxis,numpy.newaxis,:]),axis=-1)/Fosum
        sel=((r1[0:NX,1:NY+1,1:NZ+1]<r1[1:NX+1,1:NY+1,1:NZ+1])*(r1[1:NX+1,1:NY+1,1:NZ+1]>r1[2:NX+2,1:NY+1,1:NZ+1])
            *(r1[1:NX+1,0:NY,1:NZ+1]<r1[1:NX+1,1:NY+1,1:NZ+1])*(r1[1:NX+1,1:NY+1,1:NZ+1]>r1[1:NX+1,2:NY+2,1:NZ+1])
            *(r1[1:NX+1,1:NY+1,0:NZ]<r1[1:NX+1,1:NY+1,1:NZ+1])*(r1[1:NX+1,1:NY+1,1:NZ+1]>r1[1:NX+1,1:NY+1,2:NZ+2]))
        r1p=r1[1:NX+1,1:NY+1,1:NZ+1][sel]
        xp=xg[1:NX+1,1:NY+1,1:NZ+1][sel]
        yp=yg[1:NX+1,1:NY+1,1:NZ+1][sel]
        zp=zg[1:NX+1,1:NY+1,1:NZ+1][sel]
        peaks=[]
        for i in range(len(r1p)):
            peaks.append((xp[i],yp[i],zp[i],r1p[i]))
    nc=Nc 

    def refine32(x0,y0,z0,sx0,sy0,sz0,h,k,l,f2,Ah1,Bh1,Fosum,Fo,fcorrection):
        sx,sy,sz = sx0/2,sy0/2,sz0/2
        grds = {}
        for i in range(-2,3):
            for j in range(-2,3):
                for kk in range(-2,3):
                    x,y,z = x0+i*sx,y0+j*sy,z0+kk*sz 
                    grds[(i,j,kk)] = rou12(x,y,z,h,k,l,f2,Ah1,Bh1,Fosum,Fo,fcorrection)
        pks = []
        for i in range(-1,2):
            for j in range(-1,2):
                for kk in range(-1,2):
                    if grds[(i-1,j,kk)]<grds[(i,j,kk)]>grds[(i+1,j,kk)]:
                        if grds[(i,j-1,kk)]<grds[(i,j,kk)]>grds[(i,j+1,kk)]:
                            if grds[(i,j,kk-1)]<grds[(i,j,kk)]>grds[(i,j,kk+1)]:
                                pks.append((x0+i*sx,y0+j*sy,z0+kk*sz,grds[(i,j,kk)]))
        if pks:
            pks.sort(key = lambda s:-s[3])
            return pks[0]
        else: 
            return (x0,y0,z0,rou12(x0,y0,z0,h,k,l,f2,Ah1,Bh1,Fosum,Fo,fcorrection))

    peaks.sort(key = lambda s:-s[3])
    n = len(peaks)
    n_cut=5*Ntotal  # used to be 5
    #n_cut=2632 # for 1ab1 with long solvent tail
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print("n peaks = ", n, "n_cut = ", n_cut,tt)

    #n_refine= 2  #10#2
    peaks=peaks[:n_cut]
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('refine peaks... ', time.time()-starttime,tt)
    def refine_peaks2(peaks,h,k,l,f2,Ah1,Bh1,Fosum,Fo,fcorrection,na,nb,nc):
        peaks=peaks[:]
        for i in range(len(peaks)):
            sx0,sy0,sz0 = 1/na,1/nb,1/nc 
            for j in range(1):  # was 6
                x,y,z,f = peaks[i]
                peaks[i] = refine32(x,y,z,sx0,sy0,sz0,h,k,l,f2,Ah1,Bh1,Fosum,Fo,fcorrection)
                sx0,sy0,sz0 = sx0/2,sy0/2,sz0/2 
        return peaks 

    dn=int(n_cut/ncpus)+1
    n1,n2=-dn,0 
    jobs=[]
    for i in range(ncpus):
        n1,n2=n1+dn,n2+dn 
        jobs.append(job_server.submit(refine_peaks2,(peaks[n1:n2],h,k,l,f2,
            Ah1,Bh1,Fosum,Fo,fcorrection,na,nb,nc),(refine32,rou12,),
            ('numpy',),globals=globals()))

    peaks_new=[]
    for job in jobs:
        peaks_new+=job()

    peaks_new.sort(key = lambda s:-s[3])
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('Found the peaks!',tt)
    return peaks_new 





def peaks_for_globalmin_using_dips(h,k,l,Fo,A,molecule,Z,ntotal,atom_list,
    starttime=None,runs='sR1_method',more_info=None):
    # run single atom global min to build model from scratch
    # ntoal = expand to
    # atom_list = already determined part
    # molecule and Z: expected atom list 

    if starttime is None: starttime = time.time()
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('Starting global min with dips...',tt)

    F2=Fo**2

    a,b,c = abc(A)

    sl = get_sl(h,k,l,A)

    # the part of molecule already finished:
    atomj,atom_labels,solution=atomj_solution(atom_list)

    # the part not finished yet:
    content=get_content(molecule,Z)
    Natoms=0  
    for atom in content:
        Natoms+=content[atom]
    for atom in atomj:
        content[atom]-=1

    do_special=False
    if more_info is not None: # extend single atom
        try:
            next_atom,next_label,jj,d0,dd=more_info #'C','1'
            content[next_atom]-=1
            atoms,labels=atoms_labels_from_content(content)
            atoms,labels=[next_atom]+atoms,[next_label]+labels 
        except:
            try:
                next_atom,next_label,jj,nn=more_info #'C','1'
                content[next_atom]-=1
                atoms,labels=atoms_labels_from_content(content)
                atoms,labels=[next_atom]+atoms,[next_label]+labels 
            except:
                try:
                    jj,NN=more_info
                    do_special=True  
                    L=max(a,b,c)
                    na,nb,nc=int(a/0.4),int(b/0.4),int(c/0.4)
                    Na,Nb,Nc=int((2*NN+1)*a/L),int((2*NN+1)*b/L),int((2*NN+1)*c/L)
                    NC=int(Natoms*Na*Nb*Nc/na/nb/nc)
                    content['C']-=NC  
                    atoms,labels=atoms_labels_from_content(content)
                    atoms,labels=['C']*NC+atoms,['1']*NC+labels 
                except:
                    print('incorrect more_info')
                    return
    else:
        atoms,labels=atoms_labels_from_content(content)

    # the whole molecule:
    heavy_atom=atomj+atoms 
    heavy_label=atom_labels+labels  
    Nheavy=len(heavy_atom)

    # all atomic scattering factors
    f2a = {}
    for atom in heavy_atom:
        if atom not in f2a:
            f2a[atom] = numpy.array([fsc(s,atom) for s in sl]) 
    f2S = numpy.array([f2a[heavy_atom[i]]  for i in range(Nheavy)])

    # the heaviest unfinished atomic factor
    jj=len(atomj)
    f2 = f2S[jj]

    # calculate correction
    for i in range(jj,len(f2S)):
        if i == jj:
            fcorrection = f2S[i]**2
        else:
            fcorrection += f2S[i]**2

    xj = numpy.array([s[0] for s in solution])
    yj = numpy.array([s[1] for s in solution])
    zj = numpy.array([s[2] for s in solution])

    fj_tmp =f2S[:jj]
    fj=fj_tmp.T

    chj=(tpi*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
        +l[:,numpy.newaxis]*zj[numpy.newaxis,:]))
    Ahj = (fj * sin(chj) )
    Bhj = (fj * cos(chj) )
    Ah1 =numpy.sum(Ahj ,axis=-1)
    Bh1 =numpy.sum(Bhj ,axis=-1)

    iheavy=jj
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print("jj = ",jj,tt)
    with open('history.txt','a') as f:
        print("jj = ",jj,tt,file=f)

    s_precision =  0.4
    s = s_precision 

    peaks=filter(h,k,l,F2,A,heavy_atom,atom_list,starttime,more_info=more_info)
    return peaks 









# for position sym frag

def rou1a(x,y,z,h,k,l,fj,Ah1,Bh1,Fosum,Fo,fcorrection,atom_list):
    frag=add_sym_fragment((x,y,z),atom_list)
    xj = numpy.array([s[2] for s in frag])
    yj = numpy.array([s[3] for s in frag])
    zj = numpy.array([s[4] for s in frag])
    angle = 6.283185306*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
             +l[:,numpy.newaxis]*zj[numpy.newaxis,:])
    Ahj = numpy.sum(fj * numpy.sin(angle)  ,axis=-1)
    Bhj = numpy.sum(fj * numpy.cos(angle)  ,axis=-1) 
    Ah =Ah1 + Ahj
    Bh =Bh1 + Bhj 
    Fc = numpy.sqrt(Ah**2+Bh**2+fcorrection)
    r1 = (abs(Fc-Fo)).sum()/Fosum
    return -r1


def get_min_a(X,Y,Z,h,k,l,fj,fcorrection,Ah1,Bh1,Fo,Fosum,atom_list):
    r1s=[]
    for i in range(len(X)):
        x,y,z=X[i],Y[i],Z[i]
        r1=-rou1a(x,y,z,h,k,l,fj,Ah1,Bh1,Fosum,Fo,fcorrection,atom_list)
        r1s.append((r1,(x,y,z)))
    return r1s


# for position sym frag


def position_sym_frag(h,k,l,Fo,A,molecule,Z,atom_list,fragment0,
    starttime=None,runs='position_sym_frag'):
    # position a fragment which only depends on (x,y,z)
    # atom_list = already determined part
    # molecule and Z: expected atom list 
    # fragment0: list of atoms in the fragment, the coordinates are meaningless as input

    if starttime is None: starttime = time.time()
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('Starting position sym frag...',tt)

    F2=Fo**2

    a,b,c = abc(A)

    sl = get_sl(h,k,l,A)

    # the part of molecule already finished:
    atomj,atom_labels,solution=atomj_solution(atom_list)

    # the sym frag part:
    atomj_frag,labels_frag,solution_frag=atomj_solution(fragment0)

    # the part not finished yet:
    content=get_content(molecule,Z)
    for atom in atomj+atomj_frag:
        content[atom]-=1

    atoms,labels=atoms_labels_from_content(content) # this is the part not finished yet

    # the whole molecule as ordered properly
    heavy_atom=atomj+atomj_frag+atoms 
    heavy_label=atom_labels+labels_frag+labels  
    Nheavy=len(heavy_atom)

    # all atomic scattering factors
    f2a = {}
    for atom in heavy_atom:
        if atom not in f2a:
            f2a[atom] = numpy.array([fsc(s,atom) for s in sl]) 
    f2S = numpy.array([f2a[heavy_atom[i]]  for i in range(Nheavy)])

    # start and end of the sym fragment
    jj1,jj2=len(atomj),len(atomj+atomj_frag)
    
    # calculate correction
    for i in range(jj2,len(f2S)):
        if i == jj2:
            fcorrection = f2S[i]**2
        else:
            fcorrection += f2S[i]**2

    xj = numpy.array([s[0] for s in solution])
    yj = numpy.array([s[1] for s in solution])
    zj = numpy.array([s[2] for s in solution])

    fj_tmp =f2S[:jj1]
    fj=fj_tmp.T  # the known part

    Ahj = (fj * sin(tpi*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
        +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
    Bhj = (fj * cos(tpi*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
        +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
    Ah1 =numpy.sum(Ahj ,axis=-1)
    Bh1 =numpy.sum(Bhj ,axis=-1)

    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print("jj1 = ",jj1,' jj2 = ',jj2,tt)

    peaks=filter_sym_frag(h,k,l,F2,A,heavy_atom,atom_list,starttime,fragment0)

    X,Y,Z=[],[],[]
    for x,y,z,ff in peaks:
        if (not notnear3((x,y,z),solution,atomj,A)) or trianglebonding((x,y,z),solution,A):
            pass 
        else:
            X.append(x),Y.append(y),Z.append(z)
    X,Y,Z=numpy.array(X),numpy.array(Y),numpy.array(Z)



    Fosum=Fo.sum()
 
    previoustime=time.time()

    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('locating the fragment...',tt)

    fj_tmp =f2S[jj1:jj2]
    fj=fj_tmp.T   # the sym fragment

    Ntot=len(X)
    Ncut=int(Ntot/ncpus)+1
    N1,N2,jobs=0,Ncut,[] 
    while N1<Ntot:
        jobs.append(job_server.submit(get_min_a,(X[N1:N2],Y[N1:N2],Z[N1:N2],
            h,k,l,fj,fcorrection,Ah1,Bh1,Fo,Fosum,atom_list,),(rou1a,add_sym_fragment,),('numpy',)))
        N1,N2=N1+Ncut,N2+Ncut 

    r1s=[]  
    for job in jobs:
        r1s+=job()

    r1s.sort(key=lambda xx:xx[0])

    for i in range(len(r1s)):
        r1,(x,y,z)=r1s[i]
        p=(put_in_cell(x),put_in_cell(y),put_in_cell(z))
        frag=add_sym_fragment(p,atom_list)
        save_history(atom_list+frag,runs='trial '+str(i))
        try_next=input('hit enter key to try next (y/n)... ')
        if try_next=='n': break

    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('Finished position sym frag!',tt)
    return atom_list+frag 


# for position sym frag


def filter_sym_frag(h,k,l,F2,A,heavy_atom,atom_list,starttime,fragment0):
    # locate all pR1 holes 
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('Searching sym frag peaks...',tt)

    a,b,c = abc(A)

    Fo=sqrt(F2)
    Fosum=Fo.sum()

    Ntotal=len(heavy_atom)

    sl = get_sl(h,k,l,A)
    f2a = {}
    for atom in heavy_atom:
        if atom not in f2a:
            f2a[atom] = numpy.array([fsc(s,atom) for s in sl]) 
    f2S = numpy.array([f2a[heavy_atom[i]]  for i in range(Ntotal)])


    # known part
    atomj,atom_labels,solution=atomj_solution(atom_list)

    jj1,jj2 = len(atom_list),len(atom_list+fragment0) # start and end of fragment

    # calculate correction
    fcorrection=0  # need this in case there are no additional missing atoms
    for i in range(jj2,Ntotal):
        if i == jj2:
            fcorrection = f2S[i]**2
        else:
            fcorrection += f2S[i]**2


    xj = numpy.array([s[0] for s in solution])
    yj = numpy.array([s[1] for s in solution])
    zj = numpy.array([s[2] for s in solution])

    fj_tmp =f2S[:jj1]
    fj=fj_tmp.T   # for known part

    Ahj = (fj * sin(tpi*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
        +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
    Bhj = (fj * cos(tpi*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
        +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
    Ah1 =numpy.sum(Ahj ,axis=-1)
    Bh1 =numpy.sum(Bhj ,axis=-1)

    fj_tmp =f2S[jj1:jj2]
    fj=fj_tmp.T  # for sym fragment


    runs="refining"

    s=0.4
    na,nb,nc = int(a/s),int(b/s),int(c/s)
    X = numpy.array([i/na for i in range(na)])
    Y = numpy.array([i/nb for i in range(nb)])
    Z0 = numpy.array([i/nc for i in range(-1,nc+1)])


    def get_peaks1a(XX,Y,Z0,h,k,l,fj,Ah1,Bh1,Fosum,Fo,fcorrection,na,nb,nc,atom_list):
        peaks = []
        for x in XX:
            for y in Y:
                R0=numpy.array([rou1a(x,y,z,h,k,l,fj,Ah1,Bh1,Fosum,Fo,fcorrection,atom_list) for z in Z0])
                R1=R0[:nc]
                R=R0[1:nc+1]
                Z=Z0[1:nc+1]
                R2=R0[2:nc+2]
                Rs1=R[(R1<R)*(R>R2)]
                Zs1=Z[(R1<R)*(R>R2)]
                Rs1x1=numpy.array([rou1a(x-1/na,y,z,h,k,l,fj,Ah1,Bh1,Fosum,Fo,fcorrection,atom_list) for z in Zs1])
                Rs1x2=numpy.array([rou1a(x+1/na,y,z,h,k,l,fj,Ah1,Bh1,Fosum,Fo,fcorrection,atom_list) for z in Zs1])
                Rs2=Rs1[(Rs1x1<Rs1)*(Rs1>Rs1x2)]
                Zs2=Zs1[(Rs1x1<Rs1)*(Rs1>Rs1x2)]
                Rs2y1=numpy.array([rou1a(x,y-1/nb,z,h,k,l,fj,Ah1,Bh1,Fosum,Fo,fcorrection,atom_list) for z in Zs2])
                Rs2y2=numpy.array([rou1a(x,y+1/nb,z,h,k,l,fj,Ah1,Bh1,Fosum,Fo,fcorrection,atom_list) for z in Zs2])
                Rs3=Rs2[(Rs2y1<Rs2)*(Rs2>Rs2y2)]
                Zs3=Zs2[(Rs2y1<Rs2)*(Rs2>Rs2y2)]
                for i in range(len(Zs3)):
                    z,ff=Zs3[i],Rs3[i]
                    peaks.append((x,y,z,ff)) 
        return peaks 

    jobs=[]
    nX=len(X)
    dn=int(nX/ncpus)+1 
    n1,n2=-dn,0 
    for i in range(ncpus):
        n1,n2=n1+dn,n2+dn 
        jobs.append(job_server.submit(get_peaks1a,(X[n1:n2],Y,Z0,h,k,l,fj,Ah1,Bh1,
            Fosum,Fo,fcorrection,na,nb,nc,atom_list),
            (rou1a,add_sym_fragment,),('numpy','math',)))
    peaks=[]
    for job in jobs:
        peaks+=job()




    def refine3a(x0,y0,z0,sx0,sy0,sz0,h,k,l,fj,Ah1,Bh1,Fosum,Fo,fcorrection,atom_list):
        sx,sy,sz = sx0/2,sy0/2,sz0/2
        grds = {}
        for i in range(-2,3):
            for j in range(-2,3):
                for kk in range(-2,3):
                    x,y,z = x0+i*sx,y0+j*sy,z0+kk*sz 
                    grds[(i,j,kk)] = rou1a(x,y,z,h,k,l,fj,Ah1,Bh1,Fosum,Fo,fcorrection,atom_list)
        pks = []
        for i in range(-1,2):
            for j in range(-1,2):
                for kk in range(-1,2):
                    if grds[(i-1,j,kk)]<grds[(i,j,kk)]>grds[(i+1,j,kk)]:
                        if grds[(i,j-1,kk)]<grds[(i,j,kk)]>grds[(i,j+1,kk)]:
                            if grds[(i,j,kk-1)]<grds[(i,j,kk)]>grds[(i,j,kk+1)]:
                                pks.append((x0+i*sx,y0+j*sy,z0+kk*sz,grds[(i,j,kk)]))
        if pks:
            pks.sort(key = lambda s:-s[3])
            return pks[0]
        else: 
            return (x0,y0,z0,rou1a(x0,y0,z0,h,k,l,fj,Ah1,Bh1,Fosum,Fo,fcorrection,atom_list))

    peaks.sort(key = lambda s:-s[3])
    n = len(peaks)
    n_cut=n
    n_cut=5*Ntotal
    n_cut=min(40,Ntotal)  
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print("n peaks = ", n, "n_cut = ", n_cut, tt)

    #n_refine= 2  #10#2
    peaks=peaks[:n_cut]
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('refine peaks... ', time.time()-starttime,tt)

    def refine_peaks2a(peaks,h,k,l,fj,Ah1,Bh1,Fosum,Fo,fcorrection,na,nb,nc,atom_list):
        peaks=peaks[:]
        for i in range(len(peaks)):
            sx0,sy0,sz0 = 1/na,1/nb,1/nc 
            for j in range(6):
                x,y,z,f = peaks[i]
                peaks[i] = refine3a(x,y,z,sx0,sy0,sz0,h,k,l,fj,Ah1,Bh1,Fosum,Fo,fcorrection,atom_list)
                sx0,sy0,sz0 = sx0/2,sy0/2,sz0/2 
        return peaks 


    dn=int(n_cut/ncpus)+1
    n1,n2=-dn,0 
    jobs=[]
    for i in range(ncpus):
        n1,n2=n1+dn,n2+dn 
        jobs.append(job_server.submit(refine_peaks2a,(peaks[n1:n2],h,k,l,fj,
            Ah1,Bh1,Fosum,Fo,fcorrection,na,nb,nc,atom_list),(refine3a,rou1a,add_sym_fragment,),
            ('numpy',),globals=globals()))

    peaks_new=[]
    for job in jobs:
        peaks_new+=job()

    peaks_new.sort(key = lambda s:-s[3])
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('Found the peaks!',tt)
    return peaks_new 












def refine_one_fragment(h,k,l,Fo,A,molecule,Z,atom_list,fragment_i,Na,Nd,starttime=None,runs='refine one fragment'):
    # atom_list: the known partial model (could be a complete model)
    # fragment_i: list of indices that make up the fragment
    # Na steps up-down in angle, stepsize 5 degree
    # Nd steps up-down in xyz, stepsize 0.4 A
    # return improved atom_list

    if starttime is None: starttime = time.time()
    previoustime = starttime
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('refine one fragment...',tt)

    pcs,o=[],numpy.zeros(3)
    for i in range(len(fragment_i)):
        j=fragment_i[i]
        atom,label,x,y,z=atom_list[j]
        pcs.append(numpy.array((x,y,z)))
        o+=pcs[i]
    o/=len(fragment_i)
    p1=None  
    for i in range(len(pcs)):
        d=d_exact(pcs[i]-o,A)
        if d>0.2:
            p1=pcs[i]
            break
    p2=None  
    for i in range(len(pcs)):
        d1=d_exact(pcs[i]-p1,A)
        d2=d_exact(pcs[i]-o,A)
        if d1>0.2 and d2>0.2:
            p2=pcs[i]
            break

    # from now on, the local origin is o; x is from o to p1; y component from o to p2
    xp,yp,zp=local_xpypzp(o,p1,p2,A)
    # convert to local cartesians
    pcs=cells_to_local_cartesians(pcs,o,xp,yp,zp,A) # from now on pcs is in local cartesians
    atoms,labels,ps=atomj_solution(atom_list) # ps are fraction coord of the known partial structure

    F2=Fo**2

    a,b,c = abc(A)

    sl = get_sl(h,k,l,A)

    # the whole expected molecule:
    content=get_content(molecule,Z)

    # all atomic scattering factors
    f2a = {}
    for atom in content:
        if atom not in f2a:
            f2a[atom] = numpy.array([fsc(s,atom) for s in sl]) 

    # the missing part of the structure now in content
    for atom in atoms:
        content[atom]-=1

    # calculate correction
    fcorrection=0*f2a[atoms[0]]**2  
    for atom in content:
        fcorrection+=content[atom]*f2a[atom]**2

    Fosum=Fo.sum()

    fj_tmp =numpy.array([f2a[atom] for atom in atoms])
    fj=fj_tmp.T

    def tweak(psi,phi,ita,x,y,z,pcs,atom_list,fragment_i,o,xp,yp,zp): # angles in degrees
        W=getW(psi,phi,ita)
        pcs2=rotates(pcs,W)
        d=numpy.array((x,y,z))
        fp=local_cartesians_to_cells(pcs2,o+d,xp,yp,zp)
        for i in range(len(fragment_i)):
            j=fragment_i[i]
            atom,label,x,y,z=atom_list[j]
            x,y,z=fp[i]
            atom_list[j]=atom,label,x,y,z
        return atom_list

    def rou13_4(psi,phi,ita,x,y,z,pcs,atom_list,fragment_i,o,xp,yp,zp,fj,h,k,l,fcorrection,Fosum,Fo): # angles in degrees
        atom_list=tweak(psi,phi,ita,x,y,z,pcs,atom_list,fragment_i,o,xp,yp,zp)

        xj = numpy.array([s[2] for s in atom_list])
        yj = numpy.array([s[3] for s in atom_list])
        zj = numpy.array([s[4] for s in atom_list])

        Ahj = (fj * numpy.sin(6.283185306*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
            +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
        Bhj = (fj * numpy.cos(6.283185306*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
            +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
        Ah1 =numpy.sum(Ahj ,axis=-1)
        Bh1 =numpy.sum(Bhj ,axis=-1)
        Fc = numpy.sqrt(Ah1**2+Bh1**2+fcorrection)
        r1 = (abs(Fc-Fo)).sum()/Fosum
        return r1

    runs="refine one fragment"

    def collect_r1s4(cases,pcs,atom_list,fragment_i,o,xp,yp,zp,fj,h,k,l,fcorrection,Fosum,Fo):
        r1s=[]
        for psi,phi,ita,x,y,z in cases:
            r1=rou13_4(psi,phi,ita,x,y,z,pcs,atom_list,fragment_i,o,xp,yp,zp,fj,h,k,l,fcorrection,Fosum,Fo)
            r1s.append(r1)
        return r1s 

    sa,sx,sy,sz=5.0,0.4/a,0.4/b,0.4/c
    psi,phi,ita,x,y,z=0.,0.,0.,0.,0.,0.,
    #tweak orientation and position together
    if 1:
        for repeat in range(6):
            cases=[]
            if repeat:
                for ips in range(-1,2):
                    psi+=ips*sa 
                    for iph in range(-1,2):
                        phi+=iph*sa 
                        for iit in range(-1,2):
                            ita+=iit*sa 
                            for ix in range(-1,2):
                                x+=ix*sx 
                                for iy in range(-1,2):
                                    y+=iy*sy 
                                    for iz in range(-1,2):
                                        z+=iz*sz 
                                        cases.append((psi,phi,ita,x,y,z))
            else:
                for ips in range(-Na,Na+1):
                    psi+=ips*sa 
                    for iph in range(-Na,Na+1):
                        phi+=iph*sa 
                        for iit in range(-Na,Na+1):
                            ita+=iit*sa 
                            for ix in range(-Nd,Nd+1):
                                x+=ix*sx 
                                for iy in range(-Nd,Nd+1):
                                    y+=iy*sy 
                                    for iz in range(-Nd,Nd+1):
                                        z+=iz*sz 
                                        cases.append((psi,phi,ita,x,y,z))

            jobs=[]
            nX=len(cases)
            dn=int(nX/ncpus)+1 
            n1,n2=-dn,0 
            for i in range(ncpus):
                n1,n2=n1+dn,n2+dn 
                jobs.append(job_server.submit(collect_r1s4,(cases[n1:n2],pcs,atom_list,fragment_i,o,
                    xp,yp,zp,fj,h,k,l,fcorrection,Fosum,Fo),(rou13_4,tweak,getW,rotates,rotate,Wxy,Wyz,
                    local_cartesians_to_cells,local_cartesian_to_cell,),('numpy','math',)))
            r1s=[]
            for job in jobs:
                r1s+=job()

            imin=numpy.argmin(r1s)
            r1min=r1s[imin]
            psi,phi,ita,x,y,z=cases[imin]
            atom_list=tweak(psi,phi,ita,x,y,z,pcs,atom_list,fragment_i,o,xp,yp,zp)
            tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
            print(repeat,r1min,tt)
            sa,sx,sy,sz=sa/2,sx/2,sy/2,sz/2

    #tweak orientation and position separately with weak coupling
    if 0:
        for repeat in range(6):
            cases=[]
            if repeat:
                for ips in range(-2,3):
                    psi+=ips*sa 
                    for iph in range(-2,3):
                        phi+=iph*sa 
                        for iit in range(-2,23):
                            ita+=iit*sa 
                            cases.append((psi,phi,ita,x,y,z))
            else:
                for ips in range(-Na,Na+1):
                    psi+=ips*sa 
                    for iph in range(-Na,Na+1):
                        phi+=iph*sa 
                        for iit in range(-Na,Na+1):
                            ita+=iit*sa 
                            cases.append((psi,phi,ita,x,y,z))

            jobs=[]
            nX=len(cases)
            dn=int(nX/ncpus)+1 
            n1,n2=-dn,0 
            for i in range(ncpus):
                n1,n2=n1+dn,n2+dn 
                jobs.append(job_server.submit(collect_r1s4,(cases[n1:n2],pcs,atom_list,fragment_i,o,
                    xp,yp,zp,fj,h,k,l,fcorrection,Fosum,Fo),(rou13_4,tweak,getW,rotates,rotate,Wxy,Wyz,
                    local_cartesians_to_cells,local_cartesian_to_cell,),('numpy','math',)))
            r1s=[]
            for job in jobs:
                r1s+=job()

            imin=numpy.argmin(r1s)
            r1min=r1s[imin]
            psi,phi,ita,x,y,z=cases[imin]


            cases=[]
            if repeat:
                for ix in range(-2,3):
                    x+=ix*sx 
                    for iy in range(-2,3):
                        y+=iy*sy 
                        for iz in range(-2,3):
                            z+=iz*sz 
                            cases.append((psi,phi,ita,x,y,z))
            else:
                for ix in range(-Nd,Nd+1):
                    x+=ix*sx 
                    for iy in range(-Nd,Nd+1):
                        y+=iy*sy 
                        for iz in range(-Nd,Nd+1):
                            z+=iz*sz 
                            cases.append((psi,phi,ita,x,y,z))

            jobs=[]
            nX=len(cases)
            dn=int(nX/ncpus)+1 
            n1,n2=-dn,0 
            for i in range(ncpus):
                n1,n2=n1+dn,n2+dn 
                jobs.append(job_server.submit(collect_r1s4,(cases[n1:n2],pcs,atom_list,fragment_i,o,
                    xp,yp,zp,fj,h,k,l,fcorrection,Fosum,Fo),(rou13_4,tweak,getW,rotates,rotate,Wxy,Wyz,
                    local_cartesians_to_cells,local_cartesian_to_cell,),('numpy','math',)))
            r1s=[]
            for job in jobs:
                r1s+=job()

            imin=numpy.argmin(r1s)
            r1min=r1s[imin]
            psi,phi,ita,x,y,z=cases[imin]


            atom_list=tweak(psi,phi,ita,x,y,z,pcs,atom_list,fragment_i,o,xp,yp,zp)
            tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
            print(repeat,r1min,tt)
            sa,sx,sy,sz=sa/2,sx/2,sy/2,sz/2


    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('refined one fragment!',tt)
    return atom_list





def determine_fragment_by_1d_torsion(h,k,l,Fo,A,molecule,Z,atom_list,fragment_fname,p1,p2,p3,
    starttime=None,runs='determine fragment by torsion'):
    # atom_list: the known partial model
    # fragment_fname: filename of fragment data in cartesian coord: atom, label, x,y,z
    # p1,p2,p3: reference point in fraction coord
    # actual ref points use p2,2*p2-p1,p3+p2-p1
    # local origin of fragment is set at p2
    # return atom_list_fragment: in fraction coordinate

    if starttime is None: starttime = time.time()
    previoustime = starttime
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('determine fragment by 1d torsion...',tt)

    def read_cartesian(fname):
        with open(fname,'r') as f:
            text=f.read()
        lines=text.split('\n')
        atom_list_p=[]
        for line in lines:
            try:
                words=line.split()
                atom,label,x,y,z=words[0],words[1],words[2],words[3],words[4]
                x,y,z=float(x),float(y),float(z)
                atom_list_p.append((atom,label,x,y,z))
            except:
                pass 
        return atom_list_p

    p1,p2,p3=1*p2,2*p2-p1,p3+p2-p1 # from now on, the local origin is p1; x is from p1 to p2; y component from p1 to p3
    xp,yp,zp=local_xpypzp(p1,p2,p3,A)
    atom_list_p=read_cartesian(fragment_fname)
    atoms_p,labels_p,pcs=atomj_solution(atom_list_p) # pcs are cartesian coord of fragment before rotation    
    atoms,labels,ps=atomj_solution(atom_list) # ps are fraction coord of the known partial structure

    F2=Fo**2

    a,b,c = abc(A)

    sl = get_sl(h,k,l,A)

    # the whole expected molecule:
    content=get_content(molecule,Z)

    # all atomic scattering factors
    f2a = {}
    for atom in content:
        if atom not in f2a:
            f2a[atom] = numpy.array([fsc(s,atom) for s in sl]) 

    # the missing part of the structure now in content
    for atom in atoms+atoms_p:
        content[atom]-=1

    # calculate correction
    fcorrection=0*f2a[atoms[0]]**2  
    for atom in content:
        fcorrection+=content[atom]*f2a[atom]**2

    Fosum=Fo.sum()

    fj_tmp =numpy.array([f2a[atom] for atom in atoms+atoms_p])
    fj=fj_tmp.T

    def add_fragment2_1(psi,pcs,p1,xp,yp,zp):
        W=Wyz(psi)
        pcs2=rotates(pcs,W)
        fragment_ps=local_cartesians_to_cells(pcs2,p1,xp,yp,zp)
        return fragment_ps

    def rou13_3(x,ps,pcs,p1,xp,yp,zp,fj,h,k,l,fcorrection,Fosum,Fo):
        fragment_ps=add_fragment2_1(x,pcs,p1,xp,yp,zp) # x=psi in degrees

        xj = numpy.array([s[0] for s in ps+fragment_ps])
        yj = numpy.array([s[1] for s in ps+fragment_ps])
        zj = numpy.array([s[2] for s in ps+fragment_ps])

        Ahj = (fj * numpy.sin(6.283185306*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
            +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
        Bhj = (fj * numpy.cos(6.283185306*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
            +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
        Ah1 =numpy.sum(Ahj ,axis=-1)
        Bh1 =numpy.sum(Bhj ,axis=-1)
        Fc = numpy.sqrt(Ah1**2+Bh1**2+fcorrection)
        r1 = (abs(Fc-Fo)).sum()/Fosum
        return -r1

    runs="torsion"

    psi0,maxpsi=0,360.  

    cases=numpy.linspace(psi0,maxpsi,360)

    def collect_r1s2(cases,ps,pcs,p1,xp,yp,zp,fj,h,k,l,fcorrection,Fosum,Fo):
        r1s=[]
        for x in cases:
            r1=rou13_3(x,ps,pcs,p1,xp,yp,zp,fj,h,k,l,fcorrection,Fosum,Fo)
            r1s.append(r1)
        return r1s 

    jobs=[]
    nX=len(cases)
    dn=int(nX/ncpus)+1 
    n1,n2=-dn,0 
    for i in range(ncpus):
        n1,n2=n1+dn,n2+dn 
        jobs.append(job_server.submit(collect_r1s2,(cases[n1:n2],ps,pcs,p1,
            xp,yp,zp,fj,h,k,l,fcorrection,Fosum,Fo),(rou13_3,
            add_fragment2_1,Wyz,rotates,rotate,
            local_cartesians_to_cells),('numpy','math',)))
    r1s=[]
    for job in jobs:
        r1s+=job()

    # imin=numpy.argmin(r1s)
    # r1min=r1s[imin]
    # psimin=cases[imin]
    # d=(maxpsi-psi0)/100
    # cases=numpy.linspace(psimin-d,psimin+d,100)

    # jobs=[]
    # nX=len(cases)
    # dn=int(nX/ncpus)+1 
    # n1,n2=-dn,0 
    # for i in range(ncpus):
    #     n1,n2=n1+dn,n2+dn 
    #     jobs.append(job_server.submit(collect_r1s2,(cases[n1:n2],ps,pcs,p1,
    #         xp,yp,zp,fj,h,k,l,fcorrection,Fosum,Fo),(rou13_3,
    #         add_fragment2_1,Wyz,rotates,rotate,
    #         local_cartesians_to_cells),('numpy','math',)))
    # r1s=[]
    # for job in jobs:
    #     r1s+=job()

    imin=numpy.argmin(r1s)
    r1min=r1s[imin]
    psimin=cases[imin]
    fragment_ps=add_fragment2_1(psimin,pcs,p1,xp,yp,zp)


    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('determined a fragment by torsion!',tt)
    return to_atom_list(atoms_p,labels_p,fragment_ps)




def determine_fragment_by_3d_orientation(h,k,l,Fo,A,molecule,Z,atom_list,fragment_fname,p1,p2,p3,
    starttime=None,runs='determine fragment by orientation'):
    # atom_list: the known partial model
    # fragment_fname: filename of fragment data in cartesian coord: atom, label, x,y,z
    # p1,p2,p3: reference point in fraction coord
    # actual ref points use p2,2*p2-p1,p3+p2-p1
    # local origin of fragment is set at p2
    # return atom_list_fragment: in fraction coordinate

    if starttime is None: starttime = time.time()
    previoustime = starttime
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('determine fragment by 3d orientation...',tt)

    def read_cartesian(fname):
        with open(fname,'r') as f:
            text=f.read()
        lines=text.split('\n')
        atom_list_p=[]
        for line in lines:
            try:
                words=line.split()
                atom,label,x,y,z=words[0],words[1],words[2],words[3],words[4]
                x,y,z=float(x),float(y),float(z)
                atom_list_p.append((atom,label,x,y,z))
            except:
                pass 
        return atom_list_p

    p1,p2,p3=1*p2,2*p2-p1,p3+p2-p1 # from now on, the local origin is p1; x is from p1 to p2; y component from p1 to p3
    xp,yp,zp=local_xpypzp(p1,p2,p3,A)
    atom_list_p=read_cartesian(fragment_fname)
    atoms_p,labels_p,pcs=atomj_solution(atom_list_p) # pcs are cartesian coord of fragment before rotation    
    atoms,labels,ps=atomj_solution(atom_list) # ps are fraction coord of the known partial structure

    F2=Fo**2

    a,b,c = abc(A)

    sl = get_sl(h,k,l,A)

    # the whole expected molecule:
    content=get_content(molecule,Z)

    # all atomic scattering factors
    f2a = {}
    for atom in content:
        if atom not in f2a:
            f2a[atom] = numpy.array([fsc(s,atom) for s in sl]) 

    # the missing part of the structure now in content
    for atom in atoms+atoms_p:
        content[atom]-=1

    # calculate correction
    fcorrection=0*f2a[atoms[0]]**2  
    for atom in content:
        fcorrection+=content[atom]*f2a[atom]**2

    Fosum=Fo.sum()

    fj_tmp =numpy.array([f2a[atom] for atom in atoms+atoms_p])
    fj=fj_tmp.T

    def add_fragment2(psi,phi,ita,pcs,p1,xp,yp,zp):
        W=getW(psi,phi,ita)
        pcs2=rotates(pcs,W)
        fragment_ps=local_cartesians_to_cells(pcs2,p1,xp,yp,zp)
        return fragment_ps

    def rou13_1(x,y,z,ps,pcs,p1,xp,yp,zp,fj,h,k,l,fcorrection,Fosum,Fo):
        fragment_ps=add_fragment2(x,y,z,pcs,p1,xp,yp,zp) # x=psi,y=phi,z=ita in degrees

        xj = numpy.array([s[0] for s in ps+fragment_ps])
        yj = numpy.array([s[1] for s in ps+fragment_ps])
        zj = numpy.array([s[2] for s in ps+fragment_ps])

        Ahj = (fj * numpy.sin(6.283185306*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
            +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
        Bhj = (fj * numpy.cos(6.283185306*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
            +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
        Ah1 =numpy.sum(Ahj ,axis=-1)
        Bh1 =numpy.sum(Bhj ,axis=-1)
        Fc = numpy.sqrt(Ah1**2+Bh1**2+fcorrection)
        r1 = (abs(Fc-Fo)).sum()/Fosum
        return -r1

    runs="filtering"

    s=5.0
    psi0,phi0,ita0,maxpsi,maxphi,maxita=0,0,0,360.,180.,180.  # for benzene ring
    #psi0,phi0,ita0,maxpsi,maxphi,maxita=0,0,0,360.0,180.0,360.0 # general
    na,nb,nc = int(maxpsi/s),int(maxphi/s),int(maxita/s)
    X = numpy.array([psi0+i*maxpsi/na for i in range(na)])
    Y = numpy.array([phi0+i*maxphi/nb for i in range(nb)])
    Z0 = numpy.array([ita0+i*maxita/nc for i in range(-1,nc+1)])


    def get_peaks2_1(XX,Y,Z0,ps,pcs,p1,xp,yp,zp,fj,h,k,l,fcorrection,Fosum,Fo,na,nb,nc,maxpsi,maxphi):
        peaks = []
        for x in XX:
            for y in Y:
                R0=numpy.array([rou13_1(x,y,z,ps,pcs,p1,xp,yp,zp,fj,h,k,l,fcorrection,Fosum,Fo) for z in Z0])
                R1=R0[:nc]
                R=R0[1:nc+1]
                Z=Z0[1:nc+1]
                R2=R0[2:nc+2]
                Rs1=R[(R1<R)*(R>R2)]
                Zs1=Z[(R1<R)*(R>R2)]
                Rs1x1=numpy.array([rou13_1(x-maxpsi/na,y,z,ps,pcs,p1,xp,yp,zp,fj,h,k,l,fcorrection,Fosum,Fo) for z in Zs1])
                Rs1x2=numpy.array([rou13_1(x+maxpsi/na,y,z,ps,pcs,p1,xp,yp,zp,fj,h,k,l,fcorrection,Fosum,Fo) for z in Zs1])
                Rs2=Rs1[(Rs1x1<Rs1)*(Rs1>Rs1x2)]
                Zs2=Zs1[(Rs1x1<Rs1)*(Rs1>Rs1x2)]
                Rs2y1=numpy.array([rou13_1(x,y-maxphi/nb,z,ps,pcs,p1,xp,yp,zp,fj,h,k,l,fcorrection,Fosum,Fo) for z in Zs2])
                Rs2y2=numpy.array([rou13_1(x,y+maxphi/nb,z,ps,pcs,p1,xp,yp,zp,fj,h,k,l,fcorrection,Fosum,Fo) for z in Zs2])
                Rs3=Rs2[(Rs2y1<Rs2)*(Rs2>Rs2y2)]
                Zs3=Zs2[(Rs2y1<Rs2)*(Rs2>Rs2y2)]
                for i in range(len(Zs3)):
                    z,ff=Zs3[i],Rs3[i]
                    peaks.append((x,y,z,ff)) 
        return peaks 

    jobs=[]
    nX=len(X)
    dn=int(nX/ncpus)+1 
    n1,n2=-dn,0 
    for i in range(ncpus):
        n1,n2=n1+dn,n2+dn 
        jobs.append(job_server.submit(get_peaks2_1,(X[n1:n2],Y,Z0,ps,pcs,p1,xp,yp,zp,fj,h,k,l,
            fcorrection,Fosum,Fo,na,nb,nc,maxpsi,maxphi),(rou13_1,add_fragment2,getW,Wyz,Wxy,
            rotates,rotate,local_cartesians_to_cells,),
            ('numpy','math',)))
    peaks=[]
    for job in jobs:
        peaks+=job()



    def refine33_1(x0,y0,z0,sx0,sy0,sz0,ps,pcs,p1,xp,yp,zp,fj,h,k,l,fcorrection,Fosum,Fo):
        sx,sy,sz = sx0/2,sy0/2,sz0/2
        grds = {}
        for i in range(-2,3):
            for j in range(-2,3):
                for kk in range(-2,3):
                    x,y,z = x0+i*sx,y0+j*sy,z0+kk*sz 
                    grds[(i,j,kk)] = rou13_1(x,y,z,ps,pcs,p1,xp,yp,zp,fj,h,k,l,fcorrection,Fosum,Fo)
        pks = []
        for i in range(-1,2):
            for j in range(-1,2):
                for kk in range(-1,2):
                    if grds[(i-1,j,kk)]<grds[(i,j,kk)]>grds[(i+1,j,kk)]:
                        if grds[(i,j-1,kk)]<grds[(i,j,kk)]>grds[(i,j+1,kk)]:
                            if grds[(i,j,kk-1)]<grds[(i,j,kk)]>grds[(i,j,kk+1)]:
                                pks.append((x0+i*sx,y0+j*sy,z0+kk*sz,grds[(i,j,kk)]))
        if pks:
            pks.sort(key = lambda s:-s[3])
            return pks[0]
        else: 
            return (x0,y0,z0,rou13_1(x0,y0,z0,ps,pcs,p1,xp,yp,zp,fj,h,k,l,fcorrection,Fosum,Fo))

    peaks.sort(key = lambda s:-s[3])
    n = len(peaks)
    #n_cut=n
    n_cut=min(1,n)
    print("n peaks = ", n, "n_cut = ", n_cut)
    with open('history.txt','a') as f:
        print("n peaks = ", n, "n_cut = ", n_cut,file=f)

    #n_refine=6
    peaks=peaks[:n_cut]
    sx_0,sy_0,sz_0 = maxpsi/na,maxphi/nb,maxita/nc
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('refine peaks...',time.time()-starttime,tt)
    def refine_peaks3_1(peaks,ps,pcs,p1,xp,yp,zp,fj,h,k,l,fcorrection,Fosum,Fo,sx_0,sy_0,sz_0):
        peaks=peaks[:]
        for i in range(len(peaks)):
            sx0,sy0,sz0 = sx_0,sy_0,sz_0 
            for j in range(6):
                #print('j=',j)
                x,y,z,f = peaks[i]
                peaks[i] = refine33_1(x,y,z,sx0,sy0,sz0,ps,pcs,p1,xp,yp,zp,fj,h,k,l,fcorrection,Fosum,Fo)
                sx0,sy0,sz0 = sx0/2,sy0/2,sz0/2 
        return peaks 

    # dn=int(n_cut/ncpus)+1
    # n1,n2=-dn,0 
    # jobs=[]
    # for i in range(ncpus):
    #     n1,n2=n1+dn,n2+dn 
    #     print(n1,n2,n_cut)
    #     jobs.append(job_server.submit(refine_peaks3_1,(peaks[n1:n2],ps,pcs,p1,xp,yp,zp,fj,h,k,l,fcorrection,Fosum,Fo,
    #         sx_0,sy_0,sz_0),(refine33_1,rou13_1,add_fragment2,getW,Wyz,Wxy,rotates,rotate,local_cartesians_to_cells,),
    #         ('numpy','math',),globals=globals()))

    # peaks_new=[]
    # i=0
    # for job in jobs:
    #     i+=1
    #     peaks_new+=job()
    #     tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    #     print(i,ncpus,tt)

    # peaks_new.sort(key = lambda s:-s[3])
    # peaks=peaks_new[:]

    peaks=refine_peaks3_1(peaks[:1],ps,pcs,p1,xp,yp,zp,fj,h,k,l,fcorrection,Fosum,Fo,sx_0,sy_0,sz_0)
    x,y,z,ff=peaks[0]
    fragment_ps=add_fragment2(x,y,z,pcs,p1,xp,yp,zp) # x=psi,y=phi,z=ita in degrees    
    atom_list_p=to_atom_list(atoms_p,labels_p,fragment_ps)

    # x_data,y_data=[],[]
    # for i in range(len(peaks)):
    #     x_data.append(i)
    #     y_data.append(peaks[i][3])
    # from matplotlib import pyplot
    # figure=pyplot.figure(figsize=(7,7))
    # ax=figure.add_subplot(111)
    # ax.plot(x_data,y_data,'-o')
    # plt.show()

    # # save peaks
    # with open('orientations.txt','w') as f:
    #     for x,y,z,ff in peaks:
    #         print(x,y,z,ff,file=f)
    # #return

    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('determined a fragment by orientation!',tt)
    return atom_list_p




def assemble_fragments(h,k,l,Fo,A,molecule,Z,atom_list,fragment0,lines,l1,n1,
    starttime=None,runs='determine fragment by orientation'):
    # atom_list: the known partial model
    # complete the side benzene ring at n1
    # l1-n1 is the C-C bond connecting the central ring and the side ring
    # return the fragment containing the 5 C atoms that completes the side ring

    if starttime is None: starttime = time.time()
    previoustime = starttime
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('assemble fragments...',tt)

    atoms,labels,ps=atomj_solution(atom_list) # ps are fraction coord of the known partial structure

    atoms_p,labels_p,pcs=['C']*5,['1']*5,None # pcs are coord of new fragment    

    for i in range(len(ps)):
        ps[i]=numpy.array(ps[i])

    # find the temperary center position of the new ring    
    o=2*ps[n1]-ps[l1]
    # the attaching point of the new ring
    p0=ps[n1]
    # the center of the central ring
    op=2*ps[l1]-ps[n1]

    C,D=getCD(A) 

    def to_fragment(l2,o,op,p0,A,lines,D,fragment0):
        # benzene with orientation l2, bonding point at n2
        psi,phi,ita,ff=lines[l2].split()
        psi,phi,ita=float(psi),float(phi),float(ita)
        frg = add_fragment(o,psi,phi,ita,D,fragment0)
        atms,lbls,ss=atomj_solution(frg)
        for i in range(len(ss)):
            ss[i]=numpy.array(ss[i])
        # find which atom of the new ring should overlap the attaching point
        n2,dmin=None,1e100
        for i in range(len(ss)):
            p=ss[i]-op
            d=d_exact(p,A)
            if d<dmin:
                dmin,n2=d,i  
        d=p0-ss[n2]
        for i in range(len(ss)):
            ss[i]+=d 
        ss_new=[]
        for i in range(len(ss)):
            if i==n2: continue
            ss_new.append(ss[i])
        atoms_p,labels_p,pcs=['C']*5,['1']*5,None
        return to_atom_list(atoms_p,labels_p,ss_new)

    F2=Fo**2

    a,b,c = abc(A)

    sl = get_sl(h,k,l,A)

    # the whole expected molecule:
    content=get_content(molecule,Z)

    # all atomic scattering factors
    f2a = {}
    for atom in content:
        if atom not in f2a:
            f2a[atom] = numpy.array([fsc(s,atom) for s in sl]) 

    # the missing part of the structure now in content
    for atom in atoms+atoms_p:
        content[atom]-=1

    # calculate correction
    fcorrection=0*f2a[atoms[0]]**2  
    for atom in content:
        fcorrection+=content[atom]*f2a[atom]**2

    Fosum=Fo.sum()

    fj_tmp =numpy.array([f2a[atom] for atom in atoms+atoms_p])
    fj=fj_tmp.T

    def rou13_2(l2,o,op,p0,lines,D,fragment0,atom_list,A,the_heavy,fj,h,k,l,fcorrection,Fosum,Fo):
        fragment_ps=to_fragment(l2,o,op,p0,A,lines,D,fragment0)
        atoms,labels,solution=atomj_solution(atom_list)
        for atom,label,x,y,z in fragment_ps:
            p1=(x,y,z)
            if not notnear3s(p1,solution,atoms,A,the_heavy): return 1e200
            if trianglebonding(p1,solution,A): return 1e200

        xj = numpy.array([s[2] for s in atom_list+fragment_ps])
        yj = numpy.array([s[3] for s in atom_list+fragment_ps])
        zj = numpy.array([s[4] for s in atom_list+fragment_ps])

        Ahj = (fj * numpy.sin(6.283185306*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
            +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
        Bhj = (fj * numpy.cos(6.283185306*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
            +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
        Ah1 =numpy.sum(Ahj ,axis=-1)
        Bh1 =numpy.sum(Bhj ,axis=-1)
        Fc = numpy.sqrt(Ah1**2+Bh1**2+fcorrection)
        r1 = (abs(Fc-Fo)).sum()/Fosum
        return r1

    runs="assembling"

    cases=[]
    for l2 in range(len(lines)):
        cases.append(l2)

    def collect_r1s(cases,o,op,p0,lines,D,fragment0,atom_list,A,the_heavy,fj,h,k,l,fcorrection,Fosum,Fo):
        r1s=[]
        for l2 in cases:
            r1=rou13_2(l2,o,op,p0,lines,D,fragment0,atom_list,A,the_heavy,fj,h,k,l,fcorrection,Fosum,Fo)
            r1s.append(r1)
        return r1s 

    jobs=[]
    nX=len(cases)
    dn=int(nX/ncpus)+1 
    n1,n2=-dn,0 
    for i in range(ncpus):
        n1,n2=n1+dn,n2+dn 
        jobs.append(job_server.submit(collect_r1s,(cases[n1:n2],o,op,p0,lines,D,fragment0,
            atom_list,A,the_heavy,fj,h,k,l,fcorrection,Fosum,Fo),(rou13_2,to_fragment,
            add_fragment,getW,Wyz,Wxy,rotates,rotate,to_cells,to_cell,d_exact,dis_exact,
            atomj_solution,to_atom_list,notnear3s,trianglebonding,
            dis,d_min3,correct,dis_exact),('numpy','math',)))
    r1s=[]
    for job in jobs:
        r1s+=job()

    imin=numpy.argmin(r1s)
    r1min=r1s[imin]
    l2=cases[imin]
    fragment_ps=to_fragment(l2,o,op,p0,A,lines,D,fragment0)


    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('assembled fragments!',tt)
    return fragment_ps






def find_linear_orientations(h,k,l,Fo,A,molecule,Z,fragment0,
    starttime=None,runs='find linear orientations'):

    if starttime is None: starttime = time.time()
    previoustime = starttime
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('find fragment orientations...',tt)

    F2=Fo**2

    a,b,c = abc(A)

    sl = get_sl(h,k,l,A)

    # the whole expected molecule:
    content=get_content(molecule,Z)
    heavy_atom,heavy_label=atoms_labels_from_content(content)
    Nheavy=len(heavy_atom)

    # all atomic scattering factors
    f2a = {}
    for atom in heavy_atom:
        if atom not in f2a:
            f2a[atom] = numpy.array([fsc(s,atom) for s in sl]) 
    f2S = numpy.array([f2a[heavy_atom[i]]  for i in range(Nheavy)])

    # calculate full correction
    for i in range(len(f2S)):
        if i == 0:
            fcorrection = f2S[i]**2
        else:
            fcorrection += f2S[i]**2

    # adjust correction for the fragment
    for i in range(len(fragment0)):
        fcorrection-=f2a[fragment0[i][0]]**2


    Fosum=Fo.sum()

    Ntotal=len(heavy_atom)

    atomj,atom_labels,solution=atomj_solution(fragment0)

    fj_tmp =numpy.array([f2a[atom] for atom in atomj])
    fj=fj_tmp.T

    C,D=getCD(A)

    def rou13_L(x,y,D,fragment0,fj,h,k,l,fcorrection,Fosum,Fo):
        fragment=add_linear((0.3,0.3,0.3),x,y,D,fragment0) # x=theta,y=phi in degrees

        xj = numpy.array([s[2] for s in fragment])
        yj = numpy.array([s[3] for s in fragment])
        zj = numpy.array([s[4] for s in fragment])

        Ahj = (fj * numpy.sin(6.283185306*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
            +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
        Bhj = (fj * numpy.cos(6.283185306*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
            +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
        Ah1 =numpy.sum(Ahj ,axis=-1)
        Bh1 =numpy.sum(Bhj ,axis=-1)
        Fc = numpy.sqrt(Ah1**2+Bh1**2+fcorrection)
        r1 = (abs(Fc-Fo)).sum()/Fosum
        return -r1

    runs="filtering"

    s=1.0
    #maxtheta,maxphi=180.,360.  # for non-symmetric linear molecule
    maxtheta,maxphi=180.,180.  # for symmetric linear molecule
    na,nb = int(maxtheta/s),int(maxphi/s)
    X = numpy.array([i*maxtheta/na for i in range(na)])
    Y0 = numpy.array([i*maxphi/nb for i in range(-1,nb+1)])


    def get_peaks2_L(XX,Y0,D,fragment0,fj,h,k,l,fcorrection,Fosum,Fo,na,nb,maxtheta):
        peaks = []
        for x in XX:
            R0=numpy.array([rou13_L(x,y,D,fragment0,fj,h,k,l,fcorrection,Fosum,Fo) for y in Y0])
            R1=R0[:nb]
            R=R0[1:nb+1]
            Y=Y0[1:nb+1]
            R2=R0[2:nb+2]
            Rs1=R[(R1<R)*(R>R2)]
            Ys1=Y[(R1<R)*(R>R2)]
            Rs1x1=numpy.array([rou13_L(x-maxtheta/na,y,D,fragment0,fj,h,k,l,fcorrection,Fosum,Fo) for y in Ys1])
            Rs1x2=numpy.array([rou13_L(x+maxtheta/na,y,D,fragment0,fj,h,k,l,fcorrection,Fosum,Fo) for y in Ys1])
            Rs2=Rs1[(Rs1x1<Rs1)*(Rs1>Rs1x2)]
            Ys2=Ys1[(Rs1x1<Rs1)*(Rs1>Rs1x2)]
            for i in range(len(Ys2)):
                y,ff=Ys2[i],Rs2[i]
                peaks.append((x,y,ff)) 
        return peaks 

    jobs=[]
    nX=len(X)
    dn=int(nX/ncpus)+1 
    n1,n2=-dn,0 
    for i in range(ncpus):
        n1,n2=n1+dn,n2+dn 
        jobs.append(job_server.submit(get_peaks2_L,(X[n1:n2],Y0,D,fragment0,fj,h,k,l,
            fcorrection,Fosum,Fo,na,nb,maxtheta),(rou13_L,add_linear,to_cell,),
            ('numpy','math',)))
    peaks=[]
    for job in jobs:
        peaks+=job()



    def refine33_L(x0,y0,sx0,sy0,D,fragment0,fj,h,k,l,fcorrection,Fosum,Fo):
        sx,sy = sx0/2,sy0/2
        grds = {}
        for i in range(-2,3):
            for j in range(-2,3):
                x,y = x0+i*sx,y0+j*sy
                grds[(i,j)] = rou13_L(x,y,D,fragment0,fj,h,k,l,fcorrection,Fosum,Fo)
        pks = []
        for i in range(-1,2):
            for j in range(-1,2):
                if grds[(i-1,j)]<grds[(i,j)]>grds[(i+1,j)]:
                    if grds[(i,j-1)]<grds[(i,j)]>grds[(i,j+1)]:
                        pks.append((x0+i*sx,y0+j*sy,grds[(i,j)]))
        if pks:
            pks.sort(key = lambda s:-s[2])
            return pks[0]
        else: 
            return (x0,y0,rou13_L(x0,y0,D,fragment0,fj,h,k,l,fcorrection,Fosum,Fo))

    peaks.sort(key = lambda s:-s[2])
    n = len(peaks)
    #n_cut=min(1000,n)
    n_cut=n  
    n_cut=200  
    print("n peaks = ", n, "n_cut = ", n_cut)
    with open('history.txt','a') as f:
        print("n peaks = ", n, "n_cut = ", n_cut,file=f)

    #n_refine=6
    peaks=peaks[:n_cut]
    sx_0,sy_0= maxtheta/na,maxphi/nb
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('refine peaks...',time.time()-starttime,tt)
    def refine_peaks3_L(peaks,D,fragment0,fj,h,k,l,fcorrection,Fosum,Fo,sx_0,sy_0):
        peaks=peaks[:]
        for i in range(len(peaks)):
            sx0,sy0 = sx_0,sy_0 
            for j in range(6):
                x,y,f = peaks[i]
                peaks[i] = refine33_L(x,y,sx0,sy0,D,fragment0,fj,h,k,l,fcorrection,Fosum,Fo)
                sx0,sy0 = sx0/2,sy0/2
        return peaks 

    dn=int(n_cut/ncpus)+1
    n1,n2=-dn,0 
    jobs=[]
    for i in range(ncpus):
        n1,n2=n1+dn,n2+dn 
        print(n1,n2,n_cut)
        jobs.append(job_server.submit(refine_peaks3_L,(peaks[n1:n2],D,fragment0,fj,h,k,l,fcorrection,Fosum,Fo,
            sx_0,sy_0),(refine33_L,rou13_L,add_linear,to_cell,),
            ('numpy','math',),globals=globals()))

    peaks_new=[]
    i=0
    for job in jobs:
        i+=1
        peaks_new+=job()
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        print(i,ncpus,tt)

    peaks_new.sort(key = lambda s:-s[2])
    peaks=peaks_new[:]

    x_data,y_data=[],[]
    for i in range(len(peaks)):
        x_data.append(i)
        y_data.append(peaks[i][2])
    from matplotlib import pyplot
    figure=pyplot.figure(figsize=(7,7))
    ax=figure.add_subplot(111)
    ax.plot(x_data,y_data,'-o')
    plt.show()

    # save peaks
    with open('orientations_linear.txt','w') as f:
        for x,y,ff in peaks:
            print(x,y,ff,file=f)

    x,y,ff=peaks[0]
    fragment=add_linear((0.3,0.3,0.3),x,y,D,fragment0)
    save_history(fragment,runs='best orientation')

    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('Found linear orientations!',tt)
    return peaks






def find_fragment_orientations(h,k,l,Fo,A,molecule,Z,atom_list0,fragment0,n_fold,p0,
    starttime=None,runs='find fragment orientations'):

    if starttime is None: starttime = time.time()
    previoustime = starttime
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('find fragment orientations...',tt)

    F2=Fo**2

    a,b,c = abc(A)

    #sl = get_sl(h,k,l,A)

    # the whole expected molecule:
    content=get_content(molecule,Z)
    heavy_atom,heavy_label=atoms_labels_from_content(content)
    Nheavy=len(heavy_atom)

    # all atomic scattering factors
    # f2a = {}
    # for atom in heavy_atom:
    #     if atom not in f2a:
    #         f2a[atom] = numpy.array([fsc(s,atom) for s in sl]) 
    f2S = numpy.array([f2a[heavy_atom[i]]  for i in range(Nheavy)])

    # calculate full correction
    for i in range(len(f2S)):
        if i == 0:
            fcorrection = f2S[i]**2
        else:
            fcorrection += f2S[i]**2

    # adjust correction for the fragment
    for i in range(len(fragment0)):
        fcorrection-=f2a[fragment0[i][0]]**2
    for i in range(len(atom_list0)):
        fcorrection-=f2a[atom_list0[i][0]]**2


    Fosum=Fo.sum()

    Ntotal=len(heavy_atom)

    atomj,atom_labels,solution=atomj_solution(atom_list0+fragment0)

    fj_tmp =numpy.array([f2a[atom] for atom in atomj])
    fj=fj_tmp.T

    C,D=getCD(A)

    def rou13(x,y,z,D,atom_list0,fragment0,p0,fj,h,k,l,fcorrection,Fosum,Fo):
        fragment=atom_list0+add_fragment(p0,x,y,z,D,fragment0) # x=psi,y=phi,z=ita in degrees

        xj = numpy.array([s[2] for s in fragment])
        yj = numpy.array([s[3] for s in fragment])
        zj = numpy.array([s[4] for s in fragment])

        Ahj = (fj * numpy.sin(6.283185306*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
            +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
        Bhj = (fj * numpy.cos(6.283185306*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
            +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
        Ah1 =numpy.sum(Ahj ,axis=-1)
        Bh1 =numpy.sum(Bhj ,axis=-1)
        Fc = numpy.sqrt(Ah1**2+Bh1**2+fcorrection)
        r1 = (abs(Fc-Fo)).sum()/Fosum
        return -r1

    runs="filtering"

    s=5.0
    maxpsi,maxphi,maxita=360.,180.,360/n_fold  
    na,nb,nc = int(maxpsi/s),int(maxphi/s),int(maxita/s)
    X = numpy.array([i*maxpsi/na for i in range(na)])
    Y = numpy.array([i*maxphi/nb for i in range(nb)])
    Z0 = numpy.array([i*maxita/nc for i in range(-1,nc+1)])


    def get_peaks2(XX,Y,Z0,D,atom_list0,fragment0,p0,fj,h,k,l,fcorrection,Fosum,Fo,na,nb,nc,maxpsi,maxphi):
        peaks = []
        for x in XX:
            for y in Y:
                R0=numpy.array([rou13(x,y,z,D,atom_list0,fragment0,p0,fj,h,k,l,fcorrection,Fosum,Fo) for z in Z0])
                R1=R0[:nc]
                R=R0[1:nc+1]
                Z=Z0[1:nc+1]
                R2=R0[2:nc+2]
                Rs1=R[(R1<R)*(R>R2)]
                Zs1=Z[(R1<R)*(R>R2)]
                Rs1x1=numpy.array([rou13(x-maxpsi/na,y,z,D,atom_list0,fragment0,p0,fj,h,k,l,fcorrection,Fosum,Fo) for z in Zs1])
                Rs1x2=numpy.array([rou13(x+maxpsi/na,y,z,D,atom_list0,fragment0,p0,fj,h,k,l,fcorrection,Fosum,Fo) for z in Zs1])
                Rs2=Rs1[(Rs1x1<Rs1)*(Rs1>Rs1x2)]
                Zs2=Zs1[(Rs1x1<Rs1)*(Rs1>Rs1x2)]
                Rs2y1=numpy.array([rou13(x,y-maxphi/nb,z,D,atom_list0,fragment0,p0,fj,h,k,l,fcorrection,Fosum,Fo) for z in Zs2])
                Rs2y2=numpy.array([rou13(x,y+maxphi/nb,z,D,atom_list0,fragment0,p0,fj,h,k,l,fcorrection,Fosum,Fo) for z in Zs2])
                Rs3=Rs2[(Rs2y1<Rs2)*(Rs2>Rs2y2)]
                Zs3=Zs2[(Rs2y1<Rs2)*(Rs2>Rs2y2)]
                for i in range(len(Zs3)):
                    z,ff=Zs3[i],Rs3[i]
                    peaks.append((x,y,z,ff)) 
        return peaks 

    jobs=[]
    nX=len(X)
    dn=int(nX/ncpus)+1 
    n1,n2=-dn,0 
    for i in range(ncpus):
        n1,n2=n1+dn,n2+dn 
        jobs.append(job_server.submit(get_peaks2,(X[n1:n2],Y,Z0,D,atom_list0,fragment0,p0,fj,h,k,l,
            fcorrection,Fosum,Fo,na,nb,nc,maxpsi,maxphi),(rou13,add_fragment,getW,Wyz,Wxy,rotates,rotate,to_cells,to_cell,),
            ('numpy','math',)))
    peaks=[]
    for job in jobs:
        peaks+=job()



    def refine33(x0,y0,z0,sx0,sy0,sz0,D,atom_list0,fragment0,p0,fj,h,k,l,fcorrection,Fosum,Fo):
        sx,sy,sz = sx0/2,sy0/2,sz0/2
        grds = {}
        for i in range(-2,3):
            for j in range(-2,3):
                for kk in range(-2,3):
                    x,y,z = x0+i*sx,y0+j*sy,z0+kk*sz 
                    grds[(i,j,kk)] = rou13(x,y,z,D,atom_list0,fragment0,p0,fj,h,k,l,fcorrection,Fosum,Fo)
        pks = []
        for i in range(-1,2):
            for j in range(-1,2):
                for kk in range(-1,2):
                    if grds[(i-1,j,kk)]<grds[(i,j,kk)]>grds[(i+1,j,kk)]:
                        if grds[(i,j-1,kk)]<grds[(i,j,kk)]>grds[(i,j+1,kk)]:
                            if grds[(i,j,kk-1)]<grds[(i,j,kk)]>grds[(i,j,kk+1)]:
                                pks.append((x0+i*sx,y0+j*sy,z0+kk*sz,grds[(i,j,kk)]))
        if pks:
            pks.sort(key = lambda s:-s[3])
            return pks[0]
        else: 
            return (x0,y0,z0,rou13(x0,y0,z0,D,atom_list0,fragment0,p0,fj,h,k,l,fcorrection,Fosum,Fo))

    peaks.sort(key = lambda s:-s[3])
    n = len(peaks)
    #n_cut=n
    #n_cut=min(1000,n)
    n_cut=min(50,n)
    print("n peaks = ", n, "n_cut = ", n_cut)
    with open('history.txt','a') as f:
        print("n peaks = ", n, "n_cut = ", n_cut,file=f)

    #n_refine=6
    peaks=peaks[:n_cut]
    sx_0,sy_0,sz_0 = maxpsi/na,maxphi/nb,maxita/nc
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('refine peaks...',time.time()-starttime,tt)
    def refine_peaks3(peaks,D,atom_list0,fragment0,p0,fj,h,k,l,fcorrection,Fosum,Fo,sx_0,sy_0,sz_0):
        peaks=peaks[:]
        for i in range(len(peaks)):
            sx0,sy0,sz0 = sx_0,sy_0,sz_0 
            for j in range(6):
                x,y,z,f = peaks[i]
                peaks[i] = refine33(x,y,z,sx0,sy0,sz0,D,atom_list0,fragment0,p0,fj,h,k,l,fcorrection,Fosum,Fo)
                sx0,sy0,sz0 = sx0/2,sy0/2,sz0/2 
        return peaks 

    dn=int(n_cut/ncpus)+1
    n1,n2=-dn,0 
    jobs=[]
    for i in range(ncpus):
        n1,n2=n1+dn,n2+dn 
        print(n1,n2,n_cut)
        jobs.append(job_server.submit(refine_peaks3,(peaks[n1:n2],D,atom_list0,fragment0,p0,fj,h,k,l,fcorrection,Fosum,Fo,
            sx_0,sy_0,sz_0),(refine33,rou13,add_fragment,getW,Wyz,Wxy,rotates,rotate,to_cells,to_cell,),
            ('numpy','math',),globals=globals()))

    peaks_new=[]
    i=0
    for job in jobs:
        i+=1
        peaks_new+=job()
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        print(i,ncpus,tt)

    peaks_new.sort(key = lambda s:-s[3])
    peaks=peaks_new[:]

    x_data,y_data=[],[]
    for i in range(len(peaks)):
        x_data.append(i)
        y_data.append(peaks[i][3])
    from matplotlib import pyplot
    figure=pyplot.figure(figsize=(7,7))
    ax=figure.add_subplot(111)
    ax.plot(x_data,y_data,'-o')
    plt.show()

    # save peaks
    if 1:
        with open('orientations.txt','w') as f:
            for x,y,z,ff in peaks:
                print(x,y,z,ff,file=f)

    if 1:
        atomj0,atom_labels0,solution0=atomj_solution(atom_list0)
        def is_good_solution10(fragment):
            for atom,label,x,y,z in fragment:
                p1=(x,y,z)
                if not notnear3(p1,solution0,atomj0,A): return False
                if trianglebonding(p1,solution0,A): return False
            return True 
        for x,y,z,ff in peaks:
            fragment=add_fragment(p0,x,y,z,D,fragment0) # x=psi,y=phi,z=ita in degrees
            if is_good_solution10(fragment):
                save_history(atom_list0+fragment,runs='best orientation')
                break

    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('Found fragment orientations!',tt)
    return peaks



def find_sR1(h,k,l,Fo,A,molecule,Z,atom_list): # calculate sR1
    a,b,c = abc(A)
    sl = get_sl(h,k,l,A)

    # the whole expected molecule:
    content=get_content(molecule,Z)
    heavy_atom,heavy_label=atoms_labels_from_content(content)
    Nheavy=len(heavy_atom)

    # all atomic scattering factors
    # f2a = {}
    # for atom in heavy_atom:
    #     if atom not in f2a:
    #         f2a[atom] = numpy.array([fsc(s,atom) for s in sl]) 
    f2S = numpy.array([f2a[heavy_atom[i]]  for i in range(Nheavy)])

    # calculate full correction
    for i in range(len(f2S)):
        if i == 0:
            fcorrection = f2S[i]**2
        else:
            fcorrection += f2S[i]**2

    # adjust correction for the known part
    for i in range(len(atom_list)):
        fcorrection-=f2a[atom_list[i][0]]**2


    Fosum=Fo.sum()

    atomj,atom_labels,solution=atomj_solution(atom_list)

    fj_tmp =numpy.array([f2a[atom] for atom in atomj])
    fj=fj_tmp.T

    xj = numpy.array([s[2] for s in atom_list])
    yj = numpy.array([s[3] for s in atom_list])
    zj = numpy.array([s[4] for s in atom_list])

    Ahj = (fj * sin(tpi*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
        +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
    Bhj = (fj * cos(tpi*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
        +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
    Ah1 =numpy.sum(Ahj ,axis=-1)
    Bh1 =numpy.sum(Bhj ,axis=-1)
    Fc = sqrt(Ah1**2+Bh1**2+fcorrection)
    r1 = (abs(Fc-Fo)).sum()/Fosum
    return r1


def re_order():
    atom_list = read_atoms('a.res')
    atom_list.sort(key=lambda xx:-elements[xx[0]]['Z'])
    save_history(atom_list,'re-order atoms',True)


def filt_orientations(h,k,l,Fo,A,molecule,Z,fragment0,atom_list,
    s=0.4,starttime=None,runs='filt orientations'):
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('Filt orientations...',tt)

    with open('orientations.txt','r') as f:
        text=f.read()
    lines=text.split('\n')
    C,D=getCD(A)

    def get_orientation(line):
        psi,phi,ita,ff=line.split()
        psi,phi,ita,ff=float(psi),float(phi),float(ita),float(ff)
        return (psi,phi,ita)

    def get_fragment(line,p,D,fragment0):
        psi,phi,ita=get_orientation(line)
        return add_fragment(p,psi,phi,ita,D,fragment0)

    def get_unique(unique_lines,unique_fragments,lines,D,fragment0,A,ss):
        unique_lines=unique_lines[:]
        unique_fragments=unique_fragments[:]
        for i in range(len(lines)):
            if lines[i].strip()=='': break
            s1=get_fragment(lines[i],(0.,0.,0.),D,fragment0)
            Natoms=len(s1)

            a,b,c=abc(A)
            Nx,Ny,Nz=int(a/ss),int(b/ss),int(c/ss)
            M1=numpy.zeros((Nx,Ny,Nz))
            M2_new=numpy.zeros((Nx,Ny,Nz))

            for atom,label,x,y,z in s1:
                x,y,z=put_in_cell(x),put_in_cell(y),put_in_cell(z)
                nx=int(x*Nx)
                nx1,nx2=nx-1,nx+2
                ny=int(y*Ny)
                ny1,ny2=ny-1,ny+2
                nz=int(z*Nz)
                nz1,nz2=nz-1,nz+2
                M2_new[nx,ny,nz]=1
                for ix in range(nx1,nx2):
                    if ix<0:ix+=Nx 
                    if ix>=Nx:ix-=Nx 
                    for iy in range(ny1,ny2):
                        if iy<0:iy+=Ny 
                        if iy>=Ny:iy-=Ny 
                        for iz in range(nz1,nz2):
                            if iz<0:iz+=Nz 
                            if iz>=Nz:iz-=Nz 
                            M1[ix,iy,iz]=1 

            is_unique=True  
            for M2 in unique_fragments:
                Nmatch=(M1*M2).sum()
                if Nmatch==Natoms:
                    is_unique=False 
                    break
            if is_unique:
                unique_lines.append(lines[i])
                unique_fragments.append(M2_new)
        return (unique_lines,unique_fragments)


    # study why not unique
    if 0:
        f=open('history.txt','a')
        def get_W(line):
            psi,phi,ita=get_orientation(line)
            W=getW(psi,phi,ita)
            return W 
        def print_model(line):
            frag=get_fragment(line,(0.3,0.3,0.3),D,fragment0) 
            print('resi 1',file=f)
            i=0
            for atom,label,x,y,z in frag:
                i+=1
                print(atom+str(i),label,round(x,4),round(y,4),round(z,4),' 11.0 0.05',file=f)
        ss=0.25
        unique_lines,unique_fragments=[],[]
        for i in range(len(lines)):
            if lines[i].strip()=='': break
            s1=get_fragment(lines[i],(0.,0.,0.),D,fragment0)
            Natoms=len(s1)

            a,b,c=abc(A)
            Nx,Ny,Nz=int(a/ss),int(b/ss),int(c/ss)
            M1=numpy.zeros((Nx,Ny,Nz))
            M2_new=numpy.zeros((Nx,Ny,Nz))

            for atom,label,x,y,z in s1:
                x,y,z=put_in_cell(x),put_in_cell(y),put_in_cell(z)
                nx=int(x*Nx)
                nx1,nx2=nx-1,nx+2
                ny=int(y*Ny)
                ny1,ny2=ny-1,ny+2
                nz=int(z*Nz)
                nz1,nz2=nz-1,nz+2
                M2_new[nx,ny,nz]=1
                for ix in range(nx1,nx2):
                    if ix<0:ix+=Nx 
                    if ix>=Nx:ix-=Nx 
                    for iy in range(ny1,ny2):
                        if iy<0:iy+=Ny 
                        if iy>=Ny:iy-=Ny 
                        for iz in range(nz1,nz2):
                            if iz<0:iz+=Nz 
                            if iz>=Nz:iz-=Nz 
                            M1[ix,iy,iz]=1 

            is_unique=True  
            for j, M2 in enumerate(unique_fragments):
                Nmatch=(M1*M2).sum()
                if Nmatch==Natoms:
                    print('\n\none match found:',file=f)
                    print('old line:',unique_lines[j],file=f)
                    print('new line:',lines[i],file=f)
                    print('old W:',file=f)
                    print(get_W(unique_lines[j]),file=f)
                    print('new W:',file=f)
                    print(get_W(lines[i]),file=f)
                    print('old model: ',file=f)
                    print_model(unique_lines[j])
                    print('new model:',file=f)
                    print_model(lines[i])
                    print('\n\n',file=f)
                    is_unique=False 
                    break
            if is_unique:
                unique_lines.append(lines[i])
                unique_fragments.append(M2_new)
        f.close()
        with open('orientations_unique.txt','w') as f:
            for line in unique_lines:
                print(line,file=f)
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        print('saved unique orientations.',tt)
        return
    #end of study why not unique




    ss=0.25
    n=len(lines)
    dn=int(n/ncpus)+1
    n1,n2,jobs=-dn,0,[]
    for i in range(ncpus):
        n1,n2=n1+dn,n2+dn 
        jobs.append(job_server.submit(get_unique,([],[],lines[n1:n2],D,fragment0,A,ss),(get_orientation,
            get_fragment,add_fragment,abc,put_in_cell,getW,Wyz,Wxy,rotates,rotate,to_cells,
            to_cell,d_exact,dis_exact,),('numpy','math',)))

    uniques=[]
    for job in jobs:
        uniques.append(job())


    def get_unique2(unique_lines,unique_fragments,lines,D,fragment0,A,ss):
        unique_lines2=[]
        unique_fragments2=[]
        for i in range(len(lines)):
            if lines[i].strip()=='': break
            s1=get_fragment(lines[i],(0.,0.,0.),D,fragment0)
            Natoms=len(s1)

            a,b,c=abc(A)
            Nx,Ny,Nz=int(a/ss),int(b/ss),int(c/ss)
            M1=numpy.zeros((Nx,Ny,Nz))
            M2_new=numpy.zeros((Nx,Ny,Nz))

            for atom,label,x,y,z in s1:
                x,y,z=put_in_cell(x),put_in_cell(y),put_in_cell(z)
                nx=int(x*Nx)
                nx1,nx2=nx-1,nx+2
                ny=int(y*Ny)
                ny1,ny2=ny-1,ny+2
                nz=int(z*Nz)
                nz1,nz2=nz-1,nz+2
                M2_new[nx,ny,nz]=1
                for ix in range(nx1,nx2):
                    if ix<0:ix+=Nx 
                    if ix>=Nx:ix-=Nx 
                    for iy in range(ny1,ny2):
                        if iy<0:iy+=Ny 
                        if iy>=Ny:iy-=Ny 
                        for iz in range(nz1,nz2):
                            if iz<0:iz+=Nz 
                            if iz>=Nz:iz-=Nz 
                            M1[ix,iy,iz]=1 

            is_unique=True  
            for M2 in unique_fragments:
                Nmatch=(M1*M2).sum()
                if Nmatch==Natoms:
                    is_unique=False 
                    break
            if is_unique:
                unique_lines2.append(lines[i])
                unique_fragments2.append(M2_new)
        return (unique_lines+unique_lines2,unique_fragments+unique_fragments2)



    count=0 
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print(count,tt)
    while len(uniques)>1:
        if len(uniques)%2: uniques.append(([],[]))
        jobs=[]
        for i in range(0,len(uniques),2):
            jobs.append(job_server.submit(get_unique2,(uniques[i][0],
                uniques[i][1],uniques[i+1][0],D,fragment0,A,ss),(get_orientation,
                get_fragment,add_fragment,abc,put_in_cell,getW,Wyz,Wxy,rotates,
                rotate,to_cells,to_cell,d_exact,dis_exact,),('numpy','math',)))
        uniques=[]
        for job in jobs:
            uniques.append(job())
        count+=1
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        print(count,tt)


    unique_lines,unique_fragments=uniques[0]

    with open('orientations_unique.txt','w') as f:
        for line in unique_lines:
            print(line,file=f)
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('saved unique orientations.',tt)






def locate_a_batch_of_fragments(h,k,l,Fo,A,molecule,Z,fragment0,
    starttime=None,runs='locate a batch of fragments'):

    starttime=time.time()
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('locate a batch of fragments...',tt)
    F2=Fo**2

    with open('orientations_unique.txt','r') as f:
        text=f.read()
    lines=text.split('\n')[:-1]
    # print(len(lines))
    # return
    C,D=getCD(A)

    def get_orientation(line):
        psi,phi,ita,ff=line.split()
        psi,phi,ita,ff=float(psi),float(phi),float(ita),float(ff)
        return (psi,phi,ita)

    def get_fragment(line,p):
        psi,phi,ita=get_orientation(line)
        return add_fragment(p,psi,phi,ita,D,fragment0)

    start_first_fragment=0                   

    if start_first_fragment:
        # add first fragment
        psi,phi,ita=get_orientation(lines[0])
        #atom_list=add_fragment((0.3,0.3,0.3),psi,phi,ita,D,fragment0)
        atom_list=add_fragment((0.0,0.0,0.0),psi,phi,ita,D,fragment0)
    else:
        # read known part of the structure
        atom_list = read_atoms('a.res')

    save_history(atom_list,'start',True)
    #return

    # the part of molecule already finished (include the fragment we are adding):
    atomj,atom_labels,solution=atomj_solution(atom_list+fragment0)

    # the part not finished yet:
    content=get_content(molecule,Z)
    for atom in atomj:
        content[atom]-=1
    print(content)
    #return
    atoms,labels=atoms_labels_from_content(content)
    print(atoms)
    print(labels)
    #return

    # the whole molecule:
    heavy_atom=atomj+atoms 
    heavy_label=atom_labels+labels  
    Nheavy=len(heavy_atom)

    fragment=[]

    try:
        with open('fragment.txt','r') as f:
            text=f.read()
        flines=text.split('\n')
        for line in flines:
            try:
                atom,r=line.split()
                r=float(r)
                fragment.append((atom,r))
            except:
                pass 
    except:
        pass 

    orientations=list(range(len(lines)))[3:4]
    for ibenzene in range(1,2):
        peaks=filter_fragment(h,k,l,F2,A,heavy_atom,atom_list,fragment,starttime)
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        print('peaks found, time: ',len(peaks),time.time()-starttime,tt)

        r1min,xmin,ymin,zmin,imin=1e200,0,0,0,0
        orientation_set=list(set(orientations))
        print(orientation_set)
        for i in orientation_set:
            #print('try orientation ', i)
            psi,phi,ita=get_orientation(lines[i])
            (x,y,z)=locate_one_fragment(h,k,l,Fo,A,molecule,Z,fragment0,
                atom_list,psi,phi,ita,peaks,starttime,
                runs='find fragment locations')
            a_l=add_fragment((x,y,z),psi,phi,ita,D,fragment0)
            atom_list_min=atom_list+a_l
            r1=find_sR1(h,k,l,Fo,A,molecule,Z,atom_list_min)
            if r1<r1min:
                r1min,xmin,ymin,zmin,imin=r1,x,y,z,i
            tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
            print(ibenzene,i,imin,' r1=',r1,'r1min=',r1min,' time: ',time.time()-starttime,tt)
            with open('history.txt','a') as f:
                print('\n\n',file=f)
                print(ibenzene,i,imin,' r1=',r1,'r1min=',r1min,' time: ',time.time()-starttime,tt,file=f)
                ii=len(atom_list)
                for atom,label,x,y,z in a_l:
                    ii+=1
                    print(atom+str(ii),label,round(x,4),round(y,4),round(z,4),'11.0 0.05',file=f)
                print('\n\n',file=f)
        psi,phi,ita=get_orientation(lines[imin])
        atom_list+=add_fragment((xmin,ymin,zmin),psi,phi,ita,D,fragment0)
        save_history(atom_list,str(ibenzene),True)
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        print('finished fragment ',ibenzene,' at time ',time.time()-starttime,tt)
    save_history(atom_list,'final',True)
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('finished, time: ',time.time()-starttime,tt)




def is_good_solution2(p,psi,phi,ita,D,fragment0,solution0,atomj0,A,the_heavy):
    fragment=add_fragment(p,psi,phi,ita,D,fragment0) 
    for atom,label,x,y,z in fragment:
        p1=(x,y,z)
        if not notnear3s(p1,solution0,atomj0,A,the_heavy): return False
        if trianglebonding(p1,solution0,A): return False
    return True 

def rou21(x,y,z,psi,phi,ita,D,fragment0,atom_list,h,k,l,fcorrection,Fo,Fosum,fj):
    fragment=add_fragment((x,y,z),psi,phi,ita,D,fragment0) 
    atom_list1=atom_list+fragment

    xj = numpy.array([s[2] for s in atom_list1])
    yj = numpy.array([s[3] for s in atom_list1])
    zj = numpy.array([s[4] for s in atom_list1])

    Ahj = (fj * numpy.sin(6.283185306*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
        +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
    Bhj = (fj * numpy.cos(6.283185306*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
        +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
    Ah1 =numpy.sum(Ahj ,axis=-1)
    Bh1 =numpy.sum(Bhj ,axis=-1)
    Fc = numpy.sqrt(abs(Ah1**2+Bh1**2+fcorrection))
    r1 = (abs(Fc-Fo)).sum()/Fosum
    return -r1

def filtpeaks(peaks,psi,phi,ita,D,fragment0,solution0,atomj0,A,the_heavy,atom_list,h,k,l,fcorrection,Fo,Fosum,fj):
    peaks_new=[]
    for peak in peaks:
        x,y,z,ff=peak
        if is_good_solution2((x,y,z),psi,phi,ita,D,fragment0,solution0,atomj0,A,the_heavy):
            ff=rou21(x,y,z,psi,phi,ita,D,fragment0,atom_list,h,k,l,fcorrection,Fo,Fosum,fj)
            peaks_new.append((x,y,z,ff))
    return peaks_new




def locate_one_fragment(h,k,l,Fo,A,molecule,Z,fragment0,atom_list,
    psi,phi,ita,peaks,starttime=None,runs='find fragment locations'):

    if starttime is None: starttime = time.time()
    previoustime = starttime
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('locate one fragment...',tt)

    F2=Fo**2

    a,b,c = abc(A)

    sl = get_sl(h,k,l,A)

    # the whole expected molecule:
    content=get_content(molecule,Z)
    heavy_atom,heavy_label=atoms_labels_from_content(content)
    Nheavy=len(heavy_atom)

    # all atomic scattering factors
    f2a = {}
    for atom in heavy_atom:
        if atom not in f2a:
            f2a[atom] = numpy.array([fsc(s,atom) for s in sl]) 
    f2S = numpy.array([f2a[heavy_atom[i]]  for i in range(Nheavy)])

    # calculate full correction
    for i in range(len(f2S)):
        if i == 0:
            fcorrection = f2S[i]**2
        else:
            fcorrection += f2S[i]**2

    # adjust correction for the fragment and the known part
    atom_list1=atom_list+fragment0
    for i in range(len(atom_list1)):
        fcorrection-=f2a[atom_list1[i][0]]**2
    #print(fcorrection)


    Fosum=Fo.sum()

    Ntotal=len(heavy_atom)

    atomj,atom_labels,solution=atomj_solution(atom_list1)

    fj_tmp =numpy.array([f2a[atom] for atom in atomj])
    fj=fj_tmp.T

    C,D=getCD(A)

    atomj0,atom_labels0,solution0=atomj_solution(atom_list)
    def is_good_solution(p):
        fragment=add_fragment(p,psi,phi,ita,D,fragment0) 
        for atom,label,x,y,z in fragment:
            p1=(x,y,z)
            if not notnear3(p1,solution0,atomj0,A): return False
            if trianglebonding(p1,solution0,A): return False
        return True 

    def rou(x,y,z):
        fragment=add_fragment((x,y,z),psi,phi,ita,D,fragment0) 
        atom_list1=atom_list+fragment

        xj = numpy.array([s[2] for s in atom_list1])
        yj = numpy.array([s[3] for s in atom_list1])
        zj = numpy.array([s[4] for s in atom_list1])

        Ahj = (fj * sin(tpi*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
            +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
        Bhj = (fj * cos(tpi*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
            +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
        Ah1 =numpy.sum(Ahj ,axis=-1)
        Bh1 =numpy.sum(Bhj ,axis=-1)
        Fc = sqrt(abs(Ah1**2+Bh1**2+fcorrection))
        r1 = (abs(Fc-Fo)).sum()/Fosum
        return -r1

    runs="filtering"

    s=0.4
    na,nb,nc = int(a/s),int(b/s),int(c/s)


    n=len(peaks)
    dn=int(n/ncpus)+1
    n1,n2,jobs=-dn,0,[]
    for i in range(ncpus):
        n1,n2=n1+dn,n2+dn 
        jobs.append(job_server.submit(filtpeaks,(peaks[n1:n2],psi,phi,ita,D,fragment0,solution0,atomj0,A,
            the_heavy,atom_list,h,k,l,fcorrection,Fo,Fosum,fj),
            (is_good_solution2,rou21,add_fragment,notnear3s,trianglebonding,dis,d_min3,correct,dis_exact,
                getW,Wyz,Wxy,rotates,rotate,to_cells,to_cell),
            ('numpy','math')))

    peaks_new=[]
    for job in jobs:
        peaks_new+=job()  
    peaks=peaks_new[:] 

    # peaks=filtpeaks(peaks,psi,phi,ita,D,fragment0,solution0,atomj0,A,the_heavy,atom_list,h,k,l,fcorrection,Fo,Fosum,fj)

    def refine3(x0,y0,z0,sx0,sy0,sz0):
        sx,sy,sz = sx0/2,sy0/2,sz0/2
        grds = {}
        for i in range(-2,3):
            for j in range(-2,3):
                for kk in range(-2,3):
                    x,y,z = x0+i*sx,y0+j*sy,z0+kk*sz 
                    grds[(i,j,kk)] = rou(x,y,z)
        pks = []
        for i in range(-1,2):
            for j in range(-1,2):
                for kk in range(-1,2):
                    if grds[(i-1,j,kk)]<grds[(i,j,kk)]>grds[(i+1,j,kk)]:
                        if grds[(i,j-1,kk)]<grds[(i,j,kk)]>grds[(i,j+1,kk)]:
                            if grds[(i,j,kk-1)]<grds[(i,j,kk)]>grds[(i,j,kk+1)]:
                                pks.append((x0+i*sx,y0+j*sy,z0+kk*sz,grds[(i,j,kk)]))
        if pks:
            pks.sort(key = lambda s:-s[3])
            return pks[0]
        else: 
            return (x0,y0,z0,rou(x0,y0,z0))

    peaks.sort(key = lambda s:-s[3])
    n = len(peaks)
    print('npeaks = ', n)
    peaks_all=peaks[:]
    n1,n2=-1,0 
    while n1<n:
        n1,n2=n1+1,n2+1
        peaks=peaks_all[n1:n2]
        n_refine=8
        print('refine peaks...')
        for i in range(len(peaks)):
            sx0,sy0,sz0 = 1/na,1/nb,1/nc 
            for j in range(n_refine):
                x,y,z,f = peaks[i]
                peaks[i] = refine3(x,y,z,sx0,sy0,sz0)
                sx0,sy0,sz0 = sx0/2,sy0/2,sz0/2 
            tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
            print(runs,i,int(time.time()-starttime),-peaks[i][3],tt)
        peaks.sort(key = lambda s:-s[3])
        for x,y,z,ff in peaks:
            if is_good_solution((x,y,z)): 
                tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
                print('Found one fragment!',tt)
                return (x,y,z)

    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('Unable to find one fragment!',tt)
    return (x,y,z) 







def find_P4_locations(h,k,l,Fo,A,molecule,Z,fragment0,
    starttime=None,runs='find P4 locations'):

    starttime=time.time()
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('find P4 locations...',tt)
    F2=Fo**2

    C,D=getCD(A)

    atom_list = read_atoms('a.res')

    save_history(atom_list,'start',True)

    # the part of molecule already finished (include the fragment we are adding):
    atomj,atom_labels,solution=atomj_solution(atom_list+fragment0)

    # the part not finished yet:
    content=get_content(molecule,Z)
    for atom in atomj:
        content[atom]-=1
    print(content)
    #return
    atoms,labels=atoms_labels_from_content(content)
    print(atoms)
    print(labels)
    #return

    # the whole molecule:
    heavy_atom=atomj+atoms 
    heavy_label=atom_labels+labels  
    Nheavy=len(heavy_atom)

    fragment=[]

    with open('fragment_P4.txt','r') as f:
        text=f.read()
    flines=text.split('\n')
    for line in flines:
        try:
            atom,r=line.split()
            r=float(r)
            fragment.append((atom,r))
        except:
            pass 

    peaks=filter_fragment(h,k,l,F2,A,heavy_atom,atom_list,fragment,starttime)
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('peaks found, time: ',len(peaks),time.time()-starttime,tt)

    # from center of C60 to the center of a C atom is about 3.50 A
    # covalent radius of C is 0.77 A
    # so C60 moelecule is roughly a ball of radius 3.50+0.77=4.27 A
    # from center of P4 to the center of a P atom is about 1.35 A
    # covalent radius of P is 1.10 A
    # so P4 molecule is roughly a ball of radius 1.35+1.10=2.45 A
    # the C60 molecule is not touching any P4 molecule
    # so the distance from the center of a P4 molecule to the center of the C60 molecule
    # should be larger than 4.27+2.45=6.72 A
    # the distance between two P4 molecules should be larger that 2.45+2.45=4.90 A

    rmin1=6.72 
    rmin2=4.90
    peaks_raw,peaks=peaks[:],[]
    pC60=(0.0,0.0,0.0)
    P4s=[(0.57,0.29,0.52),(0.42,0.7,0.48)]
    for x,y,z,ff in peaks_raw:
        p=(x,y,z)
        is_good=True  
        r2=d_min3(p,pC60,A)
        if r2<rmin1*rmin1: is_good=False
        for p2 in P4s:
            r1=d_min3(p,p2,A)
            if r1<rmin2*rmin2: is_good=False
        if is_good: peaks.append((x,y,z,ff))

    with open('P4_locations_3.txt','w') as f:
        for x,y,z,ff in peaks:
            print(x,y,z,ff,file=f)

    for x,y,z,ff in peaks[:2]:
        atom_list+=add_fragment((x,y,z),0,0,0,D,fragment0)
    save_history(atom_list,runs='P4 locations')
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('finished, time: ',time.time()-starttime,tt)





def filter_fragment(h,k,l,F2,A,heavy_atom,atom_list,fragment,starttime):
    # locate all bottoms of the disoriented-fragment-r1 dips 
    # fragment: (atom, r to center) list
    # heavy_atom: complete model, start part is atom_list, middle is fragment
    # atom_list is the part of known atoms
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('filter fragment...',tt)

    a,b,c = abc(A)
    s=0.4
    na,nb,nc = int(a/s),int(b/s),int(c/s)

    # use this block if 
    # want to use all grid points as candidates
    if 1:
        x = numpy.array([i/na for i in range(0,na)])
        y = numpy.array([i/nb for i in range(0,nb)])
        z = numpy.array([i/nc for i in range(0,nc)])
        X,Y,Z = numpy.meshgrid(x,y,z)
        peaks=[]
        for i,x in numpy.ndenumerate(X):
            y,z=Y[i],Z[i]
            peaks.append((x,y,z,1.0))
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        print('Using all grid points!',tt)
        return peaks 

    Fo=sqrt(F2)
    Fosum=Fo.sum()

    Ntotal=len(heavy_atom)

    # sl = get_sl(h,k,l,A)
    # f2a = {}
    # for atom in heavy_atom:
    #     if atom not in f2a:
    #         f2a[atom] = numpy.array([fsc(s,atom) for s in sl]) 
    f2S = numpy.array([f2a[heavy_atom[i]]  for i in range(Ntotal)])


    # starting point
    atomj,atom_labels,solution=atomj_solution(atom_list)

    # completely disoriented fragment's scattering factor
    n1,n2=len(solution),len(fragment)
    f2=f2S[0]*0 
    for i in range(n2):
        r=fragment[i][1]
        if r<0.05:
            f2+=f2S[n1+i]
        else:
            f2+=f2S[n1+i]*gg(r,sl)
    f2*=1

    startfrom = len(solution)+len(fragment)

    # calculate correction
    fcorrection=0  
    for i in range(startfrom,Ntotal):
        if i == startfrom:
            fcorrection = f2S[i]**2
        else:
            fcorrection += f2S[i]**2


    xj = numpy.array([s[0] for s in solution])
    yj = numpy.array([s[1] for s in solution])
    zj = numpy.array([s[2] for s in solution])

    fj_tmp =f2S[:n1]
    fj=fj_tmp.T

    Ahj = (fj * sin(tpi*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
        +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
    Bhj = (fj * cos(tpi*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
        +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
    Ah1 =numpy.sum(Ahj ,axis=-1)
    Bh1 =numpy.sum(Bhj ,axis=-1)

    runs="filtering"


    def rou14(x,y,z,h,k,l,f2,Ah1,Bh1,Fosum,Fo,fcorrection):
        angle = 6.283185306*(h*x+k*y+l*z)
        Ahj = f2 * numpy.sin(angle) 
        Bhj = f2 * numpy.cos(angle) 
        Ah =Ah1 + Ahj
        Bh =Bh1 + Bhj 
        Fc = numpy.sqrt(Ah**2+Bh**2+fcorrection)
        r1 = (abs(Fc-Fo)).sum()/Fosum
        return -r1


    def rou24(x,y,z,h,k,l,f2,Ah1,Bh1,Fosum,Fo,fcorrection):
        angle = 6.283185306*(h*x+k*y+l*z)
        Ahj = f2 * numpy.sin(angle) 
        Bhj = f2 * numpy.cos(angle) 
        Ah =Ah1 + Ahj
        Bh =Bh1 + Bhj 
        Fc = numpy.sqrt(Ah**2+Bh**2+fcorrection)
        r1 = abs(Fc-Fo)/Fosum
        return -r1


    runs="refining"

    X = numpy.array([i/na for i in range(na)])
    Y = numpy.array([i/nb for i in range(nb)])
    Z0 = numpy.array([i/nc for i in range(-1,nc+1)])

    def get_peaks4(XX,Y,Z0,h,k,l,f2,Ah1,Bh1,Fosum,Fo,fcorrection,na,nb,nc):
        peaks = []
        for x in XX:
            for y in Y:
                R0=numpy.sum(rou24((x*numpy.ones_like(Z0))[:,numpy.newaxis],
                    (y*numpy.ones_like(Z0))[:,numpy.newaxis],Z0[:,numpy.newaxis],
                    h,k,l,f2,Ah1,Bh1,Fosum,Fo,fcorrection),axis=-1)
                R1=R0[:nc]
                R=R0[1:nc+1]
                Z=Z0[1:nc+1]
                R2=R0[2:nc+2]
                Rs1=R[(R1<R) * (R>R2)]
                Zs1=Z[(R1<R) * (R>R2)]
                Rs1x1=numpy.sum(rou24(((x-1/na)*numpy.ones_like(Zs1))[:,numpy.newaxis],
                    (y*numpy.ones_like(Zs1))[:,numpy.newaxis],Zs1[:,numpy.newaxis], 
                    h,k,l,f2,Ah1,Bh1,Fosum,Fo,fcorrection),axis=-1)
                Rs1x2=numpy.sum(rou24(((x+1/na)*numpy.ones_like(Zs1))[:,numpy.newaxis],
                    (y*numpy.ones_like(Zs1))[:,numpy.newaxis],Zs1[:,numpy.newaxis], 
                    h,k,l,f2,Ah1,Bh1,Fosum,Fo,fcorrection),axis=-1)
                Rs2=Rs1[(Rs1x1<Rs1) * (Rs1>Rs1x2)]
                Zs2=Zs1[(Rs1x1<Rs1) * (Rs1>Rs1x2)]
                Rs2y1=numpy.sum(rou24((x*numpy.ones_like(Zs2))[:,numpy.newaxis],
                    ((y-1/nb)*numpy.ones_like(Zs2))[:,numpy.newaxis],Zs2[:,numpy.newaxis], 
                    h,k,l,f2,Ah1,Bh1,Fosum,Fo,fcorrection),axis=-1)
                Rs2y2=numpy.sum(rou24((x*numpy.ones_like(Zs2))[:,numpy.newaxis],
                    ((y+1/nb)*numpy.ones_like(Zs2))[:,numpy.newaxis],Zs2[:,numpy.newaxis],
                    h,k,l,f2,Ah1,Bh1,Fosum,Fo,fcorrection),axis=-1)
                Rs3=Rs2[(Rs2y1<Rs2) * (Rs2>Rs2y2)]
                Zs3=Zs2[(Rs2y1<Rs2) * (Rs2>Rs2y2)]
                for i in range(len(Zs3)):
                    z,ff=Zs3[i],Rs3[i]
                    peaks.append((x,y,z,ff)) 
        return peaks 

    jobs=[]
    nX=len(X)
    dn=int(nX/ncpus)+1 
    n1,n2=-dn,0 
    for i in range(ncpus):
        n1,n2=n1+dn,n2+dn 
        jobs.append(job_server.submit(get_peaks4,(X[n1:n2],Y,Z0,h,k,l,f2,Ah1,Bh1,
            Fosum,Fo,fcorrection,na,nb,nc),
            (rou24,),('numpy','math',)))
    peaks=[]
    for job in jobs:
        peaks+=job()



    def refine34(x0,y0,z0,sx0,sy0,sz0,h,k,l,f2,Ah1,Bh1,Fosum,Fo,fcorrection):
        sx,sy,sz = sx0/2,sy0/2,sz0/2
        grds = {}
        for i in range(-2,3):
            for j in range(-2,3):
                for kk in range(-2,3):
                    x,y,z = x0+i*sx,y0+j*sy,z0+kk*sz 
                    grds[(i,j,kk)] = rou14(x,y,z,h,k,l,f2,Ah1,Bh1,Fosum,Fo,fcorrection)
        pks = []
        for i in range(-1,2):
            for j in range(-1,2):
                for kk in range(-1,2):
                    if grds[(i-1,j,kk)]<grds[(i,j,kk)]>grds[(i+1,j,kk)]:
                        if grds[(i,j-1,kk)]<grds[(i,j,kk)]>grds[(i,j+1,kk)]:
                            if grds[(i,j,kk-1)]<grds[(i,j,kk)]>grds[(i,j,kk+1)]:
                                pks.append((x0+i*sx,y0+j*sy,z0+kk*sz,grds[(i,j,kk)]))
        if pks:
            pks.sort(key = lambda s:-s[3])
            return pks[0]
        else: 
            return (x0,y0,z0,rou14(x0,y0,z0,h,k,l,f2,Ah1,Bh1,Fosum,Fo,fcorrection))

    peaks.sort(key = lambda s:-s[3])
    n = len(peaks)
    n_cut=max(int(5*Ntotal/len(fragment)),300)
    print("n peaks = ", n, "n_cut = ", n_cut)

    #n_refine=2
    peaks=peaks[:n_cut]
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('refine peaks... ', time.time()-starttime,tt)
    def refine_peaks4(peaks,h,k,l,f2,Ah1,Bh1,Fosum,Fo,fcorrection,na,nb,nc):
        peaks=peaks[:]
        for i in range(len(peaks)):
            sx0,sy0,sz0 = 1/na,1/nb,1/nc 
            for j in range(2):
                x,y,z,f = peaks[i]
                peaks[i] = refine34(x,y,z,sx0,sy0,sz0,h,k,l,f2,Ah1,Bh1,Fosum,Fo,fcorrection)
                sx0,sy0,sz0 = sx0/2,sy0/2,sz0/2 
        return peaks 

    dn=int(n_cut/ncpus)+1
    n1,n2=-dn,0 
    jobs=[]
    for i in range(ncpus):
        n1,n2=n1+dn,n2+dn 
        jobs.append(job_server.submit(refine_peaks4,(peaks[n1:n2],h,k,l,f2,
            Ah1,Bh1,Fosum,Fo,fcorrection,na,nb,nc),(refine34,rou14,),
            ('numpy',),globals=globals()))

    peaks_new=[]
    for job in jobs:
        peaks_new+=job()

    peaks_new.sort(key = lambda s:-s[3])
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('filtered fragments!',tt)
    return peaks_new 





def is_good_solution2_L(p,theta,phi,D,fragment0,solution0,atomj0,A,the_heavy):
    fragment=add_linear(p,theta,phi,D,fragment0) 
    for atom,label,x,y,z in fragment:
        p1=(x,y,z)
        if not notnear3s(p1,solution0,atomj0,A,the_heavy): return False
        if trianglebonding(p1,solution0,A): return False
    return True 

def rou21_L(x,y,z,theta,phi,D,fragment0,atom_list,h,k,l,fcorrection,Fo,Fosum,fj):
    fragment=add_linear((x,y,z),theta,phi,D,fragment0) 
    atom_list1=atom_list+fragment

    xj = numpy.array([s[2] for s in atom_list1])
    yj = numpy.array([s[3] for s in atom_list1])
    zj = numpy.array([s[4] for s in atom_list1])

    Ahj = (fj * numpy.sin(6.283185306*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
        +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
    Bhj = (fj * numpy.cos(6.283185306*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
        +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
    Ah1 =numpy.sum(Ahj ,axis=-1)
    Bh1 =numpy.sum(Bhj ,axis=-1)
    Fc = numpy.sqrt(abs(Ah1**2+Bh1**2+fcorrection))
    r1 = (abs(Fc-Fo)).sum()/Fosum
    return -r1



def filtpeaks_L(peaks,theta,phi,D,fragment0,solution0,atomj0,A,the_heavy,atom_list,h,k,l,fcorrection,Fo,Fosum,fj):
    peaks_new=[]
    for peak in peaks:
        x,y,z,ff=peak
        if is_good_solution2_L((x,y,z),theta,phi,D,fragment0,solution0,atomj0,A,the_heavy):
            ff=rou21_L(x,y,z,theta,phi,D,fragment0,atom_list,h,k,l,fcorrection,Fo,Fosum,fj)
            peaks_new.append((x,y,z,ff))
    return peaks_new



def locate_a_batch_of_linear_fragments(h,k,l,Fo,A,molecule,Z,fragment0,
    starttime=None,runs='locate a batch of linear fragments'):

    starttime=time.time()
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('locate a batch of linear fragments...',tt)
    F2=Fo**2

    with open('orientations_linear.txt','r') as f:
        text=f.read()
    lines=text.split('\n')[:-1]
    C,D=getCD(A)

    def get_orientation_L(line):
        theta,phi,ff=line.split()
        theta,phi,ff=float(theta),float(phi),float(ff)
        return (theta,phi)

    def get_fragment_L(line,p):
        theta,phi=get_orientation_L(line)
        return add_linear(p,theta,phi,D,fragment0)

    start_first_fragment=0                       

    if start_first_fragment:
        # add first fragment
        theta,phi=get_orientation_L(lines[0])
        atom_list=add_linear((0.3,0.3,0.3),theta,phi,D,fragment0)
    else:
        # read known part of the structure
        atom_list = read_atoms('a.res')

    save_history(atom_list,'start',True)
    #return

    # the part of molecule already finished:
    atomj,atom_labels,solution=atomj_solution(atom_list+fragment0)

    # the part not finished yet:
    content=get_content(molecule,Z)
    for atom in atomj:
        content[atom]-=1
    print(content)
    #return
    atoms,labels=atoms_labels_from_content(content)
    print(atoms)
    print(labels)
    #return

    # the whole molecule:
    heavy_atom=atomj+atoms 
    heavy_label=atom_labels+labels  
    Nheavy=len(heavy_atom)

    a,b,c = abc(A)
    s=0.4
    na,nb,nc = int(a/s),int(b/s),int(c/s)
    x = numpy.array([i/na for i in range(0,na)])
    y = numpy.array([i/nb for i in range(0,nb)])
    z = numpy.array([i/nc for i in range(0,nc)])
    X,Y,ZZ = numpy.meshgrid(x,y,z)
    peaks_std=[]
    for i,x in numpy.ndenumerate(X):
        y,z=Y[i],ZZ[i]
        peaks_std.append((x,y,z,1.0))
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('Using all grid points!',tt)

    orientations=list(range(len(lines)))[3:4]
    for ibenzene in range(0,1):
        peaks=peaks_std[:]
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        print('peaks found, time: ',len(peaks),time.time()-starttime,tt)

        r1min,xmin,ymin,zmin,imin=1e200,0,0,0,0
        orientation_set=list(set(orientations))
        print(orientation_set)
        for i in orientation_set:
            #print('try orientation ', i)
            theta,phi=get_orientation_L(lines[i])
            (x,y,z)=locate_one_linear_fragment(h,k,l,Fo,A,molecule,Z,fragment0,
                atom_list,theta,phi,peaks,starttime,
                runs='find linear fragment locations')
            a_l=add_linear((x,y,z),theta,phi,D,fragment0)
            atom_list_min=atom_list+a_l
            r1=find_sR1(h,k,l,Fo,A,molecule,Z,atom_list_min)
            if r1<r1min:
                r1min,xmin,ymin,zmin,imin=r1,x,y,z,i
            tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
            print(ibenzene,i,imin,' r1=',r1,'r1min=',r1min,' time: ',time.time()-starttime,tt)
            with open('history.txt','a') as f:
                print('\n\n',file=f)
                print(ibenzene,i,imin,' r1=',r1,'r1min=',r1min,' time: ',time.time()-starttime,tt,file=f)
                ii=len(atom_list)
                for atom,label,x,y,z in a_l:
                    ii+=1
                    print(atom+str(ii),label,round(x,4),round(y,4),round(z,4),'11.0 0.05',file=f)
                print('\n\n',file=f)
        theta,phi=get_orientation_L(lines[imin])
        atom_list+=add_linear((xmin,ymin,zmin),theta,phi,D,fragment0)
        save_history(atom_list,str(ibenzene),True)
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        print('finished fragment ',ibenzene,' at time ',time.time()-starttime,tt)
    save_history(atom_list,'final',True)
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('finished, time: ',time.time()-starttime,tt)





def locate_one_linear_fragment(h,k,l,Fo,A,molecule,Z,fragment0,atom_list,
    theta,phi,peaks,starttime=None,runs='find linear fragment locations'):

    if starttime is None: starttime = time.time()
    previoustime = starttime
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('locate one linear fragment...',tt)

    F2=Fo**2

    a,b,c = abc(A)

    #sl = get_sl(h,k,l,A)

    # the whole expected molecule:
    content=get_content(molecule,Z)
    heavy_atom,heavy_label=atoms_labels_from_content(content)
    Nheavy=len(heavy_atom)

    # all atomic scattering factors
    # f2a = {}
    # for atom in heavy_atom:
    #     if atom not in f2a:
    #         f2a[atom] = numpy.array([fsc(s,atom) for s in sl]) 
    f2S = numpy.array([f2a[heavy_atom[i]]  for i in range(Nheavy)])

    # calculate full correction
    for i in range(len(f2S)):
        if i == 0:
            fcorrection = f2S[i]**2
        else:
            fcorrection += f2S[i]**2

    # adjust correction for the fragment and the known part
    atom_list1=atom_list+fragment0
    for i in range(len(atom_list1)):
        fcorrection-=f2a[atom_list1[i][0]]**2
    #print(fcorrection)


    Fosum=Fo.sum()

    Ntotal=len(heavy_atom)

    atomj,atom_labels,solution=atomj_solution(atom_list1)

    fj_tmp =numpy.array([f2a[atom] for atom in atomj])
    fj=fj_tmp.T

    C,D=getCD(A)

    atomj0,atom_labels0,solution0=atomj_solution(atom_list)
    def is_good_solution_L(p):
        fragment=add_linear(p,theta,phi,D,fragment0) 
        for atom,label,x,y,z in fragment:
            p1=(x,y,z)
            if not notnear3(p1,solution0,atomj0,A): return False
            if trianglebonding(p1,solution0,A): return False
        return True 

    def rou_L(x,y,z):
        fragment=add_linear((x,y,z),theta,phi,D,fragment0) 
        atom_list1=atom_list+fragment

        xj = numpy.array([s[2] for s in atom_list1])
        yj = numpy.array([s[3] for s in atom_list1])
        zj = numpy.array([s[4] for s in atom_list1])

        Ahj = (fj * sin(tpi*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
            +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
        Bhj = (fj * cos(tpi*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
            +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
        Ah1 =numpy.sum(Ahj ,axis=-1)
        Bh1 =numpy.sum(Bhj ,axis=-1)
        Fc = sqrt(abs(Ah1**2+Bh1**2+fcorrection))
        r1 = (abs(Fc-Fo)).sum()/Fosum
        return -r1

    runs="filtering"

    s=0.4
    na,nb,nc = int(a/s),int(b/s),int(c/s)


    n=len(peaks)
    dn=int(n/ncpus)+1
    n1,n2,jobs=-dn,0,[]
    for i in range(ncpus):
        n1,n2=n1+dn,n2+dn 
        jobs.append(job_server.submit(filtpeaks_L,(peaks[n1:n2],theta,phi,D,fragment0,solution0,atomj0,A,
            the_heavy,atom_list,h,k,l,fcorrection,Fo,Fosum,fj),
            (is_good_solution2_L,rou21_L,add_linear,notnear3s,trianglebonding,dis,d_min3,correct,dis_exact,
                to_cell),
            ('numpy','math')))

    peaks_new=[]
    for job in jobs:
        peaks_new+=job()  
    peaks=peaks_new[:] 


    def refine3_L(x0,y0,z0,sx0,sy0,sz0):
        sx,sy,sz = sx0/2,sy0/2,sz0/2
        grds = {}
        for i in range(-2,3):
            for j in range(-2,3):
                for kk in range(-2,3):
                    x,y,z = x0+i*sx,y0+j*sy,z0+kk*sz 
                    grds[(i,j,kk)] = rou_L(x,y,z)
        pks = []
        for i in range(-1,2):
            for j in range(-1,2):
                for kk in range(-1,2):
                    if grds[(i-1,j,kk)]<grds[(i,j,kk)]>grds[(i+1,j,kk)]:
                        if grds[(i,j-1,kk)]<grds[(i,j,kk)]>grds[(i,j+1,kk)]:
                            if grds[(i,j,kk-1)]<grds[(i,j,kk)]>grds[(i,j,kk+1)]:
                                pks.append((x0+i*sx,y0+j*sy,z0+kk*sz,grds[(i,j,kk)]))
        if pks:
            pks.sort(key = lambda s:-s[3])
            return pks[0]
        else: 
            return (x0,y0,z0,rou_L(x0,y0,z0))

    peaks.sort(key = lambda s:-s[3])
    n = len(peaks)
    print('npeaks = ', n)
    peaks_all=peaks[:]
    n1,n2=-1,0 
    while n1<n:
        n1,n2=n1+1,n2+1
        peaks=peaks_all[n1:n2]
        n_refine=8
        print('refine peaks...')
        for i in range(len(peaks)):
            sx0,sy0,sz0 = 1/na,1/nb,1/nc 
            for j in range(n_refine):
                x,y,z,f = peaks[i]
                peaks[i] = refine3_L(x,y,z,sx0,sy0,sz0)
                sx0,sy0,sz0 = sx0/2,sy0/2,sz0/2 
            tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
            print(runs,i,int(time.time()-starttime),-peaks[i][3],tt)
        peaks.sort(key = lambda s:-s[3])
        for x,y,z,ff in peaks:
            if is_good_solution_L((x,y,z)): 
                tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
                print('Found one fragment!',tt)
                return (x,y,z)

    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('Unable to find one fragment!',tt)
    return (x,y,z) 









def rou15(x,y,z,dF,h,k,l,phi):
    return (dF *numpy.cos(6.283185306 * (h*x+k*y+l*z-phi))).sum()

def rou25(x,y,z,dF,h,k,l,phi):
    return dF *numpy.cos(6.283185306 * (h*x+k*y+l*z-phi))

def refine35(x0,y0,z0,sx0,sy0,sz0,dF,h,k,l,phi):
    sx,sy,sz = sx0/2,sy0/2,sz0/2
    grds = {}
    for i in range(-2,3):
        for j in range(-2,3):
            for kk in range(-2,3):
                x,y,z = x0+i*sx,y0+j*sy,z0+kk*sz 
                grds[(i,j,kk)] = rou15(x,y,z,dF,h,k,l,phi)
    pks = []
    for i in range(-1,2):
        for j in range(-1,2):
            for kk in range(-1,2):
                if grds[(i-1,j,kk)]<grds[(i,j,kk)]>grds[(i+1,j,kk)]:
                    if grds[(i,j-1,kk)]<grds[(i,j,kk)]>grds[(i,j+1,kk)]:
                        if grds[(i,j,kk-1)]<grds[(i,j,kk)]>grds[(i,j,kk+1)]:
                            pks.append((x0+i*sx,y0+j*sy,z0+kk*sz,grds[(i,j,kk)]))
    if pks:
        pks.sort(key = lambda s:-s[3])
        return pks[0]
    else: 
        return (x0,y0,z0,rou15(x0,y0,z0,dF,h,k,l,phi))

def refine_peaks25(peaks,dF,h,k,l,phi,na,nb,nc):
    peaks=peaks[:]
    for i in range(len(peaks)):
        sx0,sy0,sz0 = 1/na,1/nb,1/nc 
        for j in range(5):
            x,y,z,f = peaks[i]
            peaks[i] = refine35(x,y,z,sx0,sy0,sz0,dF,h,k,l,phi)
            sx0,sy0,sz0 = sx0/2,sy0/2,sz0/2 
    return peaks 



def get_peaks15(XX,Y,Z0,dF,h,k,l,phi,na,nb,nc):
    peaks = []
    for x in XX:
        for y in Y:
            R0=numpy.sum(rou25((x*numpy.ones_like(Z0))[:,numpy.newaxis],
                (y*numpy.ones_like(Z0))[:,numpy.newaxis],Z0[:,numpy.newaxis],
                dF,h,k,l,phi),axis=-1)
            R1=R0[:nc]
            R=R0[1:nc+1]
            Z=Z0[1:nc+1]
            R2=R0[2:nc+2]
            Rs1=R[(R1<R) * (R>R2)]
            Zs1=Z[(R1<R) * (R>R2)]
            Rs1x1=numpy.sum(rou25(((x-1/na)*numpy.ones_like(Zs1))[:,numpy.newaxis],
                (y*numpy.ones_like(Zs1))[:,numpy.newaxis],Zs1[:,numpy.newaxis], 
                dF,h,k,l,phi),axis=-1)
            Rs1x2=numpy.sum(rou25(((x+1/na)*numpy.ones_like(Zs1))[:,numpy.newaxis],
                (y*numpy.ones_like(Zs1))[:,numpy.newaxis],Zs1[:,numpy.newaxis], 
                dF,h,k,l,phi),axis=-1)
            Rs2=Rs1[(Rs1x1<Rs1) * (Rs1>Rs1x2)]
            Zs2=Zs1[(Rs1x1<Rs1) * (Rs1>Rs1x2)]
            Rs2y1=numpy.sum(rou25((x*numpy.ones_like(Zs2))[:,numpy.newaxis],
                ((y-1/nb)*numpy.ones_like(Zs2))[:,numpy.newaxis],Zs2[:,numpy.newaxis], 
                dF,h,k,l,phi),axis=-1)
            Rs2y2=numpy.sum(rou25((x*numpy.ones_like(Zs2))[:,numpy.newaxis],
                ((y+1/nb)*numpy.ones_like(Zs2))[:,numpy.newaxis],Zs2[:,numpy.newaxis],
                dF,h,k,l,phi),axis=-1)
            Rs3=Rs2[(Rs2y1<Rs2) * (Rs2>Rs2y2)]
            Zs3=Zs2[(Rs2y1<Rs2) * (Rs2>Rs2y2)]
            for i in range(len(Zs3)):
                z,ff=Zs3[i],Rs3[i]
                peaks.append((x,y,z,ff)) 
    return peaks 


# find corrected-peaks - using electron density and FFT
def find_corrected_peaks2(h,k,l,F2,A,atom_list,heavy_atom,heavy_label,Nheavy,
    Ntotal,runs=1,n=1,U=0.00,starttime=None,s=0.4,n_refine=3,cutlimit=1.5, 
    do_copy=False,mB=1, patterson=False):
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('dual space cycling...',tt)
    # dual-space cycling using FFT
    h,k,l,F2 = h.copy(),k.copy(),l.copy(),F2.copy()
    if starttime is None: starttime = time.time()
    nelectrons = elements[heavy_atom[0]]['Z']
    atom_list.sort(key=lambda a:-elements[a[0]]['Z'])
    atomj,atom_labels,solution = atomj_solution(atom_list)
    n1,n2 = len(atomj),len(heavy_atom)
    N0=n1 
    atomj = heavy_atom[:n1]+atomj[n2:]
    atom_labels = heavy_label[:n1]+atom_labels[n2:]
    atomj_input = atomj.copy()

    # final list of atoms
    atomj = atomj[:Ntotal]
    atom_labels = atom_labels[:Ntotal]
    for i in range(len(atomj),Ntotal):
        atomj.append('C')
        atom_labels.append('1')
    nn = min(len(atomj),len(heavy_atom))
    for i in range(nn):
        atomj[i] = heavy_atom[i]
        atom_labels[i]=heavy_label[i]    


    a,b,c = abc(A)

    hh,kk,ll,FF2 = h.copy(),k.copy(),l.copy(),F2.copy()
    h,k,l,F2 = kill_half_hkl(h,k,l,F2)
    sl = get_sl(h,k,l,A)
    if patterson:
        B=0.0
        E2 = F2 *exp(B*sl*sl) # version of E2 after killing half hkl

    F2o = F2.copy()
    F2o[F2o<0] = 0
    Fo = sqrt(F2o)

    # real work starts here
    xj = numpy.array([s[0] for s in solution])
    yj = numpy.array([s[1] for s in solution])
    zj = numpy.array([s[2] for s in solution])

    f2a = {}
    for atom in atomj_input:
        if atom not in f2a:
            f2a[atom] = numpy.array([fsc(s,atom) for s in sl]) 
    fj_tmp =numpy.array([f2a[atomj_input[i]]  for i in range(len(atomj_input))])
    fj=fj_tmp.T

    chj=(tpi*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
        +l[:,numpy.newaxis]*zj[numpy.newaxis,:]))
    Ahj = (fj * sin(chj) )
    Bhj = (fj * cos(chj) )
    Ah1 =numpy.sum(Ahj ,axis=-1)
    Bh1 =numpy.sum(Bhj ,axis=-1)

    F2c = Ah1**2+Bh1**2
    Fc = sqrt(F2c)

    Bh1[Bh1==0]=1e-100
    phi = atan(Ah1/Bh1)/tpi
    phi[Bh1<0] += 0.5  

    # this is the starting atomic model in the electron density point of view:
    def roue(x,y,z):
        return (Fc *cos(tpi * (h*x+k*y+l*z-phi))).sum()*2

    hmax = max(hh.max(),-hh.min())
    kmax = max(kk.max(),-kk.min())
    lmax = max(ll.max(),-ll.min())
    Nh,Nk,Nl = 2*hmax,2*kmax,2*lmax

    F2o = FF2.copy()
    F2min = abs(F2o)[F2o!=0].min()/10000
    fc = numpy.ones((Nh,Nk,Nl))
    fo = numpy.zeros((Nh,Nk,Nl))
    dc = numpy.zeros((Nh,Nk,Nl))
    mk = -numpy.ones((Nh,Nk,Nl))

    for i in range(len(hh)):
        ih,ik,il = hh[i],kk[i],ll[i]
        if ih<0: ih += Nh  
        if ik<0: ik += Nk  
        if il<0: il += Nl  
        if F2o[i]>0:
            fo[ih,ik,il] = sqrt(F2o[i])
        else:
            fo[ih,ik,il] = sqrt(F2min)
        mk[ih,ik,il] = n # n=1 as function input

    if patterson==1:
        Eo = numpy.zeros(len(E2))
        Eo[E2>=0]=sqrt(E2[E2>=0])
        Eo[E2<0]=0 
        # this is patterson function:
        def roup(x,y,z):
            return (Eo *cos(tpi * (h*x+k*y+l*z))).sum()*2
        pts=[(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]
        for ih in range(Nh):
            for ik in range(Nk):
                for il in range(Nl):
                    x,y,z = ih/Nh,ik/Nk,il/Nl
                    if notnear((x,y,z),pts,1.3,A):
                        dc[ih,ik,il] = roup(x,y,z)
                    else:
                        dc[ih,ik,il] = 0
        ih,ik,il = numpy.unravel_index(numpy.argmax(dc,axis=None),dc.shape)
        x,y,z = ih/Nh,ik/Nk,il/Nl
        grds = numpy.zeros((3,3,3))
        sx,sy,sz = 1/Nh,1/Nk,1/Nl 
        for j in range(3):
            sx,sy,sz=sx/2,sy/2,sz/2
            for ih in range(3):
                for ik in range(3):
                    for il in range(3):
                        x0,y0,z0 = x+(ih-1)*sx,y+(ik-1)*sy,z+(il-1)*sz 
                        grds[ih,ik,il]=roup(x0,y0,z0)
            ih,ik,il = numpy.unravel_index(numpy.argmax(grds,axis=None),grds.shape) 
            x,y,z = x+(ih-1)*sx,y+(ik-1)*sy,z+(il-1)*sz  
        xm,ym,zm = x,y,z 
        # this is patterson superposition min function
        def roupm(x,y,z):
            return min(roup(x,y,z),roup(x+xm,y+ym,z+zm))
        rou0 = roupm
    else:
        rou0 = roue



    for ih in range(Nh):
        for ik in range(Nk):
            for il in range(Nl):
                dc[ih,ik,il] = rou0(ih/Nh,ik/Nk,il/Nl)

    # print(Nh,Nk,Nl)
    # print(int(a/0.4),int(b/0.4),int(c/0.4))

    angle_old = numpy.angle(fc)
    old_collection, new_collection = [], []
    repeats = 0 
    fosum = (fo**2).sum()
    N0-=5
    cnt = 0
    while True:
        cnt += 1
        dc_new = dc*0
        new_collection = []
        if N0<len(atomj): N0+=5
        solution,nsolution = [],0
        for atom in atomj[:N0]:
            ih,ik,il = numpy.unravel_index(numpy.argmax(dc,axis=None),dc.shape)
            #dmax=numpy.abs(dc[ih,ik,il])
            new_collection.append((ih,ik,il))
            x,y,z = ih/Nh,ik/Nk,il/Nl

            if atom in the_heavy:
                dd=2.2
            else:
                dd=1.2
            Nh1,Nh2 = int(Nh*(x-dd/a)-1),int(Nh*(x+dd/a)+2)
            Nk1,Nk2 = int(Nk*(y-dd/b)-1),int(Nk*(y+dd/b)+2)
            Nl1,Nl2 = int(Nl*(z-dd/c)-1),int(Nl*(z+dd/c)+2)
            mask1 = numpy.ones((Nh,Nk,Nl))
            mask2 = numpy.zeros((Nh,Nk,Nl))
            for ih in range(Nh1,Nh2):
                if ih<0: ih+=Nh  
                if ih>=Nh: ih-=Nh 
                for ik in range(Nk1,Nk2):
                    if ik<0: ik+=Nk  
                    if ik>=Nk: ik-=Nk 
                    for il in range(Nl1,Nl2):
                        if il<0: il+=Nl 
                        if il>=Nl: il-=Nl 
                        if length((x,y,z),(ih/Nh,ik/Nk,il/Nl),A)<dd:
                            mask1[ih,ik,il] = 0.0 
                            dx,dy,dz = (x-ih/Nh),(y-ik/Nk),(z-il/Nl)
                            #mask2[ih,ik,il] += 1/(1+(dx*dx+dy*dy+dz*dz)/0.09)
                            mask2[ih,ik,il] += exp(-(dx*dx+dy*dy+dz*dz)/0.12) #*elements[atom]['Z']/dmax
            dc_new += dc*mask2
            dc *= mask1
        dc = dc_new
        fc = ifftn(dc)
        rf = numpy.absolute(fc)
        th2 = (numpy.angle(fc)).copy()
        dth = ((th2-angle_old)**2).sum()/Nh/Nk/Nl 
        if (dth<0.0005 and old_collection==new_collection):
            repeats+=1
        else:
            repeats = 0
        if repeats>2 or cnt>20: break
        angle_old = th2.copy()
        old_collection=new_collection  
        if cnt%100==0: print(cnt,int(time.time()-starttime),dth)
        scale = sqrt((rf**2).sum()/fosum)
        rf = (n+1)*fo-mk*rf/scale
        if patterson==2:
            patterson=0 
            rf=1.0*fo  
            #th2=numpy.random.rand(Nh,Nk,Nl)*2*numpy.pi 
            th2=numpy.linspace(0.,2*numpy.pi,Nh*Nk*Nl).reshape((Nh,Nk,Nl))+0*numpy.pi/2
            #th2=numpy.ones_like(th2)*100*numpy.pi/180 
        rf[rf<0]=0  
        fc = rf*(cos(th2)+1j*sin(th2))
        dc = fftn(fc)

    # print('total dual space cycles = ', cnt,int(time.time()-starttime), 'dth = ', dth)
    # with open('history.txt','a') as f:
    #     print('total dual space cycles = ',int(time.time()-starttime), cnt,file=f)

    th = numpy.angle(fc)/tpi 
    rf = numpy.absolute(fc)

    neg = False
    for i in range(len(h)):
        ih,ik,il = h[i],k[i],l[i]
        if ih<0 or ik<0 or il<0:neg=True  
        if ih<0: ih += Nh  
        if ik<0: ik += Nk  
        if il<0: il += Nl 
        phi[i] = th[ih,ik,il]
        Fc[i] = rf[ih,ik,il]
    # if neg: print('yes, negatives')

    scale = sqrt((Fc**2).sum()/(Fo**2).sum())


    # recalculate Fc and phi, construct final atomic model
    n1 = n+1
    dF = n1*Fo-n*Fc/scale




    runs="refining"

    s=0.4
    na,nb,nc = int(a/s),int(b/s),int(c/s)
    X = numpy.array([i/na for i in range(na)])
    Y = numpy.array([i/nb for i in range(nb)])
    Z0 = numpy.array([i/nc for i in range(-1,nc+1)])

    jobs=[]
    nX=len(X)
    dn=int(nX/ncpus)+1 
    n1,n2=-dn,0 
    for i in range(ncpus):
        n1,n2=n1+dn,n2+dn 
        jobs.append(job_server.submit(get_peaks15,(X[n1:n2],Y,Z0,dF,h,k,l,phi,na,nb,nc),
            (rou25,),('numpy','math',)))
    peaks=[]
    for job in jobs:
        peaks+=job()



    peaks.sort(key = lambda s:-s[3])
    n = len(peaks)
    #n_cut=n
    n_cut=int(cutlimit*Ntotal)+1
    print("n peaks = ", n, "n_cut = ", n_cut)

    #n_refine= 2  #10#2
    peaks=peaks[:n_cut]
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('refine peaks... ', time.time()-starttime,tt)

    dn=int(n_cut/ncpus)+1
    n1,n2=-dn,0 
    jobs=[]
    for i in range(ncpus):
        n1,n2=n1+dn,n2+dn 
        jobs.append(job_server.submit(refine_peaks25,(peaks[n1:n2],dF,h,k,l,phi,na,nb,nc),(refine35,rou15,),
            ('numpy',),globals=globals()))

    peaks_new=[]
    for job in jobs:
        peaks_new+=job()

    peaks_new.sort(key = lambda s:-s[3])
    peaks=peaks_new[:]



    x,y,z,f = peaks[0]  
    x = put_in_cell(x)
    y = put_in_cell(y)
    z = put_in_cell(z)
    peaks1 = peaks[1:]
    peaks = [(x,y,z,f)]
    for pk1 in peaks1:
        is_new = True
        x1,y1,z1,f1 = pk1
        x1 = put_in_cell(x1)
        y1 = put_in_cell(y1)
        z1 = put_in_cell(z1)
        for pk in peaks:
            x,y,z,f = pk  
            dx,dy,dz = x-x1,y-y1,z-z1
            X = [dx,dy,dz]
            if dis_exact(X,A)<1.44:  # r < 1.2
                is_new = False
                break
        if is_new : peaks.append((x1,y1,z1,f1))


    # save peaks
    num = 0
    text = ''
    ff0 = peaks[0][3]
    atomj = atomj[:Ntotal]
    atom_labels = atom_labels[:Ntotal]
    for i in range(len(atomj),Ntotal):
        atomj.append('C')
        atom_labels.append('1')
    nn = min(len(atomj),len(heavy_atom))
    for i in range(nn):
        atomj[i] = heavy_atom[i]
        atom_labels[i]=heavy_label[i]

    # remove ghost peaks, remove triangle bonding peaks
    pks = peaks[1:]
    peaks = [peaks[0]]
    x,y,z,ff = peaks[0]
    solution = [(x,y,z)]
    for x,y,z,ff in pks:
        n = len(solution)
        if n>=nn: break
        if notnear3((x,y,z),solution,atomj[:n],A):
            if not trianglebonding((x,y,z),solution,A):
                solution.append((x,y,z))
                peaks.append((x,y,z,ff))

    if True:
        for x,y,z,ff in peaks[:Ntotal]:
            num += 1
            q = st3(atomj[num-1]+str(num))+atom_labels[num-1]
            xt = st2(round(x+0,4))
            yt = st2(round(y+0,4))
            zt = st2(round(z+0,4))
            st = '  11.00 10.05  '
            ff *= nelectrons/ff0
            line = q+xt+yt+zt+st+str(round(ff,2)) 
            text += line+'\n'
    #print('the corrected-peaks are saved')
    if do_copy:cp(text[:-1])

    solution = []
    for x,y,z,ff in peaks[:Ntotal]:
        solution.append((x,y,z))
    n = len(solution)
    atomj = atomj[:n]
    atom_labels = atom_labels[:n]
    n = len(atomj)
    solution = solution[:n]
    solution = do_arrange(solution,A)
    atom_list = to_atom_list(atomj,atom_labels,solution)
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('total time: ',runs,int(time.time()-starttime),tt)
    with open('history.txt','a') as f:
        print('total time: ',runs,int(time.time()-starttime),tt,file=f)
    return atom_list 








def shift_fragment(p,fragment0):
    dx,dy,dz=p 
    fragment=[]
    for i in range(len(fragment0)):
        atom,label,x,y,z=fragment0[i]
        x,y,z=x+dx,y+dy,z+dz
        fragment.append((atom,label,x,y,z))
    return fragment

def is_good_solution35(p,fragment0,solution0,atomj0,A,the_heavy):
    fragment=shift_fragment(p,fragment0) 
    for atom,label,x,y,z in fragment:
        p1=(x,y,z)
        if not notnear3s(p1,solution0,atomj0,A,the_heavy): return False
        if trianglebonding(p1,solution0,A): return False
    return True 

def rou35(x,y,z,fragment0,atom_list,h,k,l,fcorrection,Fo,Fosum,fj):
    fragment=shift_fragment((x,y,z),fragment0) 
    atom_list1=atom_list+fragment

    xj = numpy.array([s[2] for s in atom_list1])
    yj = numpy.array([s[3] for s in atom_list1])
    zj = numpy.array([s[4] for s in atom_list1])

    Ahj = (fj * numpy.sin(6.283185306*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
        +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
    Bhj = (fj * numpy.cos(6.283185306*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
        +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
    Ah1 =numpy.sum(Ahj ,axis=-1)
    Bh1 =numpy.sum(Bhj ,axis=-1)
    Fc = numpy.sqrt(abs(Ah1**2+Bh1**2+fcorrection))
    r1 = (abs(Fc-Fo)).sum()/Fosum
    return -r1

def filtpeaks35(peaks,fragment0,solution0,atomj0,A,the_heavy,atom_list,h,k,l,fcorrection,Fo,Fosum,fj):
    peaks_new=[]
    for peak in peaks:
        x,y,z,ff=peak
        if is_good_solution35((x,y,z),fragment0,solution0,atomj0,A,the_heavy):
            ff=rou35(x,y,z,fragment0,atom_list,h,k,l,fcorrection,Fo,Fosum,fj)
            peaks_new.append((x,y,z,ff))
    return peaks_new

def placing_fragment(h,k,l,Fo,A,molecule,Z,fragment0_file,atom_list,
    starttime=None,runs='placing a fragment'):
    # the frag already in fractional coordinates with correct orientation
    # only need to shift its location
    if starttime is None: starttime = time.time()
    previoustime = starttime
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('placing a fragment...',tt)

    with open(fragment0_file,'r') as f:
        text=f.read()
    lines=text.split('\n')
    fragment0=[]
    for line in lines:
        try:
            words=line.strip().split() 
            atom,label,x,y,z=words
            x,y,z=float(x),float(y),float(z)
            fragment0.append((atom,label,x,y,z))
        except:
            pass 


    a,b,c = abc(A)

    ss=0.4
    na,nb,nc=int(a/ss),int(b/ss),int(c/ss)
    x = numpy.array([i/na for i in range(0,na)])
    y = numpy.array([i/nb for i in range(0,nb)])
    z = numpy.array([i/nc for i in range(0,nc)])
    X,Y,Zz = numpy.meshgrid(x,y,z)
    peaks=[]
    for i,x in numpy.ndenumerate(X):
        y,z=Y[i],Zz[i]
        peaks.append((x,y,z,1.0))
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('Using all grid points!',tt)

    F2=Fo**2

    #sl = get_sl(h,k,l,A)

    # the whole expected molecule:
    content=get_content(molecule,Z)
    heavy_atom,heavy_label=atoms_labels_from_content(content)
    Nheavy=len(heavy_atom)

    # all atomic scattering factors
    # f2a = {}
    # for atom in heavy_atom:
    #     if atom not in f2a:
    #         f2a[atom] = numpy.array([fsc(s,atom) for s in sl]) 
    f2S = numpy.array([f2a[heavy_atom[i]]  for i in range(Nheavy)])

    # calculate full correction
    for i in range(len(f2S)):
        if i == 0:
            fcorrection = f2S[i]**2
        else:
            fcorrection += f2S[i]**2

    # adjust correction for the fragment and the known part
    atom_list1=atom_list+fragment0
    for i in range(len(atom_list1)):
        fcorrection-=f2a[atom_list1[i][0]]**2
    #print(fcorrection)


    Fosum=Fo.sum()

    Ntotal=len(heavy_atom)

    atomj,atom_labels,solution=atomj_solution(atom_list1)

    fj_tmp =numpy.array([f2a[atom] for atom in atomj])
    fj=fj_tmp.T

    C,D=getCD(A)

    atomj0,atom_labels0,solution0=atomj_solution(atom_list)
    def is_good_solution20(p):
        fragment=shift_fragment(p,fragment0) 
        for atom,label,x,y,z in fragment:
            p1=(x,y,z)
            if not notnear3(p1,solution0,atomj0,A): return False
            if trianglebonding(p1,solution0,A): return False
        return True 

    def rou20(x,y,z):
        fragment=shift_fragment((x,y,z),fragment0) 
        atom_list1=atom_list+fragment

        xj = numpy.array([s[2] for s in atom_list1])
        yj = numpy.array([s[3] for s in atom_list1])
        zj = numpy.array([s[4] for s in atom_list1])

        Ahj = (fj * sin(tpi*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
            +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
        Bhj = (fj * cos(tpi*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
            +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
        Ah1 =numpy.sum(Ahj ,axis=-1)
        Bh1 =numpy.sum(Bhj ,axis=-1)
        Fc = sqrt(abs(Ah1**2+Bh1**2+fcorrection))
        r1 = (abs(Fc-Fo)).sum()/Fosum
        return -r1

    runs="filtering"

    s=0.4
    na,nb,nc = int(a/s),int(b/s),int(c/s)


    n=len(peaks)
    dn=int(n/ncpus)+1
    n1,n2,jobs=-dn,0,[]
    for i in range(ncpus):
        n1,n2=n1+dn,n2+dn 
        jobs.append(job_server.submit(filtpeaks35,(peaks[n1:n2],fragment0,solution0,atomj0,A,
            the_heavy,atom_list,h,k,l,fcorrection,Fo,Fosum,fj),
            (is_good_solution35,rou35,shift_fragment,notnear3s,trianglebonding,dis,d_min3,
                correct,dis_exact,),
            ('numpy','math')))

    peaks_new=[]
    for job in jobs:
        peaks_new+=job() 
    for i in range(len(peaks_new)):
        x,y,z,ff=peaks_new[i]
        ff=-rou20(x,y,z)
        peaks_new[i]=(x,y,z,ff) 
    peaks=peaks_new[:] 

    def refine45(x0,y0,z0,sx0,sy0,sz0):
        sx,sy,sz = sx0/2,sy0/2,sz0/2
        grds = {}
        for i in range(-2,3):
            for j in range(-2,3):
                for kk in range(-2,3):
                    x,y,z = x0+i*sx,y0+j*sy,z0+kk*sz 
                    grds[(i,j,kk)] = rou20(x,y,z)
        pks = []
        for i in range(-1,2):
            for j in range(-1,2):
                for kk in range(-1,2):
                    if grds[(i-1,j,kk)]<grds[(i,j,kk)]>grds[(i+1,j,kk)]:
                        if grds[(i,j-1,kk)]<grds[(i,j,kk)]>grds[(i,j+1,kk)]:
                            if grds[(i,j,kk-1)]<grds[(i,j,kk)]>grds[(i,j,kk+1)]:
                                pks.append((x0+i*sx,y0+j*sy,z0+kk*sz,grds[(i,j,kk)]))
        if pks:
            pks.sort(key = lambda s:-s[3])
            return pks[0]
        else: 
            return (x0,y0,z0,rou20(x0,y0,z0))

    peaks.sort(key = lambda s:-s[3])
    n = len(peaks)
    print('npeaks = ', n)
    peaks_all=peaks[:]
    n1,n2=-1,0 
    atom_list_solution=[]
    while n1<n:
        n1,n2=n1+1,n2+1
        peaks=peaks_all[n1:n2]
        n_refine=8
        print('refine peaks...')
        for i in range(len(peaks)):
            sx0,sy0,sz0 = 1/na,1/nb,1/nc 
            for j in range(n_refine):
                x,y,z,f = peaks[i]
                peaks[i] = refine45(x,y,z,sx0,sy0,sz0)
                sx0,sy0,sz0 = sx0/2,sy0/2,sz0/2 
            tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
            print(runs,i,int(time.time()-starttime),-peaks[i][3],tt)
        peaks.sort(key = lambda s:-s[3])
        solution_found=False
        for x,y,z,ff in peaks:
            if is_good_solution20((x,y,z)): 
                tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
                print('Found one fragment!',tt)
                solution_found=True
                fragment=shift_fragment((x,y,z),fragment0) 
                atom_list_solution=atom_list+fragment
                break
        if solution_found: break

    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    if atom_list_solution:
        save_history(atom_list_solution,runs='placing a fragment')
        print('the fragment has been placed',tt)
    else:
        print('Unable to place the fragment!',tt)












def save_history(atom_list,runs,do_copy=True):
    from datetime import datetime
    num = 0
    letters=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w',
             'x','y','z']
    mix=[]
    for i in range(10):
        for l in letters:
            mix.append(str(i)+l)
    i_a=0
    old_atom='q'
    # text = ''
    model_lines = []
    with open('history.txt','a') as f:
        f.write('\n\n\n\n\n'+str(datetime.now())+'\n')
        f.write('run number: '+str(runs)+'\n\n')
        for a,l,x,y,z in atom_list:
            l=label_dic[a]
            if a==old_atom:
                i_a+=1  
            else:
                i_a=0
                old_atom=a  
            num += 1
            if len(a+str(num))<5:
                old_atom='q'
                str_num=str(num)
            else:
                str_num=mix[i_a]
            q = st3(a+str_num)+l
            xt = st2(round(x+0,4))
            yt = st2(round(y+0,4))
            zt = st2(round(z+0,4))
            st = '  11.00 0.05  '
            line = q+xt+yt+zt+st+'\n'
            f.write(line)
            # text += line 
            model_lines.append(line.strip())
        f.write('\n\n\n\n')
        # if do_copy:
        #     cp(text)
    res_lines=res_start_lines+model_lines+['','']+res_end_lines
    with open('a.res','w') as f:
        for l in res_lines:
            print(l,file=f)


def generate_random_model():
    atom_list = read_atoms('a.res')
    atomj,atom_labels,solution=atomj_solution(atom_list)
    for i in range(len(solution)):
        solution[i] = (random(),random(),random())
    atom_list = to_atom_list(atomj,atom_labels,solution)
    save_history(atom_list,runs='random model',do_copy=True)
    return atom_list

def do_arrange(solution, A):
    pts=solution[:]
    N=len(pts)
    if N<2: return solution 
    for i in range(len(pts)):
        x,y,z=pts[i]
        x,y,z=put_in_cell(x),put_in_cell(y),put_in_cell(z)
        pts[i]=(x,y,z)
    ds=numpy.zeros((N,N))
    imin,jmin,Dmin=None,None,1e100
    for i in range(0,N-1):
        for j in range(i+1,N):
            p1,p2=pts[i],pts[j]
            dmin,p2p=d_min4(p1,p2,A)
            ds[i,j]=dmin 
            ds[j,i]=dmin
            if dmin<Dmin: 
                imin,jmin,Dmin=i,j,dmin

    candidates=list(range(N))

    p1,p2=pts[imin],pts[jmin]
    dmin,p2p=d_min4(p1,p2,A)
    pts[imin]=numpy.array(p1)
    pts[jmin]=p2p
    selected=[imin,jmin]
    candidates.remove(imin)
    candidates.remove(jmin)

    while candidates:
        imin,jmin,Dmin=None,None,1e100
        for i in selected:
            for j in candidates:
                if ds[i,j]<Dmin:
                    imin,jmin,Dmin=i,j,ds[i,j]
        selected.append(jmin)
        p1,p2=pts[imin],pts[jmin]
        dmin,p2p=d_min4(p1,p2,A)
        pts[jmin]=p2p
        candidates.remove(jmin)
    return pts 

def do_center(solution):
    pts=solution[:]
    N = len(pts)
    P=numpy.zeros(3)
    for p in pts:
        P+=p  
    P/=N 
    PC=0.5*numpy.ones(3)
    for i in range(N):
        pts[i]+=PC-P
    return pts 

def re_arrange():
    atom_list = read_atoms('a.res')
    atomj,atom_labels,solution=atomj_solution(atom_list)
    for i in range(len(solution)):
        x,y,z=solution[i]
        x,y,z=put_in_cell(x),put_in_cell(y),put_in_cell(z)
        print(x,y,z)
        solution[i]=(x,y,z)
    A = matrix_A('a.res')
    solution=do_arrange(solution,A)
    atom_list = to_atom_list(atomj,atom_labels,solution)
    save_history(atom_list,runs='re-arranged model',do_copy=True)


def cacl_r1(atomj,solution,h,k,l,F2,A): 
    F2 = F2.copy()
    sl = get_sl(h,k,l,A)

    contents = {}
    for atom in atomj:
        if atom not in contents:
            contents[atom]=1 
        else:
            contents[atom]+=1
    scale = scaling_factor(contents,sl,h,k,l,F2)

    F2 *= scale 

    xj = numpy.array([s[0] for s in solution])
    yj = numpy.array([s[1] for s in solution])
    zj = numpy.array([s[2] for s in solution])

    fj = numpy.array([[fsc(s,am) for am in atomj] for s in sl])

    Ahj = (fj * sin(tpi*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
        +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
    Bhj = (fj * cos(tpi*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
        +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
    Ah =numpy.sum(Ahj ,axis=-1)
    Bh =numpy.sum(Bhj ,axis=-1)
    F2o=F2.copy()
    F2o[F2o<0]=0.0
    total = (fj**2).sum()
    total_obs = F2o.sum()
    scale = total/total_obs
    F2o *= scale

    F2o=F2.copy()
    F2o[F2o<0]=0
    D=sqrt(Ah**2+Bh**2)-sqrt(F2o)
    return (abs(D)).sum()/sqrt(F2o).sum()

def get_r1():
    h,k,l,F2,sigF2 = read_hkl3('a.hkl')
    A = matrix_A('a.res')
    atom_list = read_atoms('a.res')
    atomj,atom_labels,solution = atomj_solution(atom_list)
    r1 = cacl_r1(atomj,solution,h,k,l,F2,A)
    print('r1 = ', r1)
    return r1 


def simu_hkl():
    h,k,l,F2,sigF2 = read_hkl3('a.hkl')
    A = matrix_A('a.res')
    atom_list = read_atoms('a.res')
    atomj,atom_labels,solution = atomj_solution(atom_list)
    r1 = cacl_r1(atomj,solution,h,k,l,F2,A)
    print('r1 = ', r1)

    #sl = get_sl(h,k,l,A)

    contents = {}
    for atom in atomj:
        if atom not in contents:
            contents[atom]=1 
        else:
            contents[atom]+=1

    scale = scaling_factor(contents,sl,h,k,l,F2)

    xj = numpy.array([s[0] for s in solution])
    yj = numpy.array([s[1] for s in solution])
    zj = numpy.array([s[2] for s in solution])

    fj = numpy.array([[fsc(s,am) for am in atomj] for s in sl])

    Ahj = (fj * sin(tpi*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
        +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
    Bhj = (fj * cos(tpi*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
        +l[:,numpy.newaxis]*zj[numpy.newaxis,:])) )
    Ah =numpy.sum(Ahj ,axis=-1)
    Bh =numpy.sum(Bhj ,axis=-1)

    F2c=(Ah**2+Bh**2)/scale  
    with open('a_simu.hkl', 'w') as f:
        for i in range(len(h)):
            F2c[i]=round(F2c[i],2)
            sigF2[i]=round(sigF2[i],2)
            f.write(st1(h[i])+st1(k[i])+st1(l[i])+st2(F2c[i])+st2(sigF2[i])+'\n')
        f.write(st1(0)+st1(0)+st1(0)+st2(0)+st2(0)+'\n')
    print('simu hkl data saved')






def find_pairs(i_models,i_corrects,atom_list,atom_list_correct,A):
    pairs=[]
    for i_model in i_models:
        atom,label,x,y,z=atom_list[i_model]
        p_model=(x,y,z)
        dmin=1000000.0
        pair=None  
        for i_correct in i_corrects:
            atom,label,x,y,z=atom_list_correct[i_correct]
            p_correct=(x,y,z)
            d=d_min5(p_correct,p_model,A)
            if d<dmin:
                dmin=d 
                pair=(i_model,i_correct,dmin)
        pairs.append(pair)
    return pairs 


def invert_model(res='a.res'):
    # read the model
    atom_list = read_atoms(res)
    for i in range(len(atom_list)):
        atom,label,x,y,z=atom_list[i]
        atom_list[i]=(atom,label,-x,-y,-z)
    save_history(atom_list,'inverted model',do_copy=True)
    print('model has been inverted')

def pairing2(correct_res='correct.res',res='a.res', s=0.5):
    # read correct model
    atom_list_correct = read_atoms(correct_res)
    # read the model
    atom_list = read_atoms(res)
    # check if same number of atoms
    if len(atom_list_correct)!=len(atom_list):
        print('not same number of atoms, please check!')
        print('# atoms in correct model = ',len(atom_list_correct),'# atoms in the model = ',len(atom_list))
        with open('history.txt','a') as f:
            print('not same number of atoms, please check!',file=f)
            print('# atoms in correct model = ',len(atom_list_correct),'# atoms in the model = ',len(atom_list),file=f)

    if len(atom_list_correct)>len(atom_list):
        Ntarget=len(atom_list)
    else:
        Ntarget=len(atom_list_correct)

    to_overlaps=[]
    for i in range(len(atom_list)):
        for j in range(len(atom_list_correct)):
            to_overlaps.append((i,j))

    A = matrix_A(correct_res)

    if 0:
        N_best_match,best_overlap=158,(0,0)
    else:
        best_overlap,N_best_match=None,0
        i_current=0
        i_skip=None  
        while to_overlaps:
            i,j=to_overlaps.pop(0)
            if i==i_skip: continue
            print((i,j),end='   ')
            if i_current==i:
                i_current+=1
                count=1
                for i_model,i_correct in to_overlaps:
                    if i_model==i: count+=1  
                if count<=N_best_match: 
                    print('')
                    i_skip=i  
                    continue  
            atom,label,xi,yi,zi=atom_list[i]
            atom,label,xj,yj,zj=atom_list_correct[j]
            dx,dy,dz=xj-xi,yj-yi,zj-zi
            atom_list_shifted=[]
            for atom,label,x,y,z in atom_list:
                atom_list_shifted.append((atom,label,x+dx,y+dy,z+dz))
            N_match=0
            i_corrects=list(range(len(atom_list_correct)))
            for i_model in range(len(atom_list_shifted)):
                atom,label,x,y,z=atom_list_shifted[i_model]
                p_model=(x,y,z)
                dmin=1000000.0
                i_correct_matched=None 
                for i_correct in i_corrects:
                    atom,label,x,y,z=atom_list_correct[i_correct]
                    p_correct=(x,y,z)
                    d=d_min5(p_correct,p_model,A)
                    if d<dmin:
                        dmin=d 
                        i_correct_matched=i_correct
                if dmin<s:
                    N_match+=1
                    try:
                        to_overlaps.remove((i_model,i_correct_matched))
                    except:
                        pass 
            if N_match>N_best_match:
                N_best_match=N_match
                best_overlap=(i,j)
                if N_best_match==Ntarget: break
            print('N_best_match = ',N_best_match,'   best overlap = ',best_overlap)
    with open('history.txt','a') as f:
        print('N_best_match = ',N_best_match,file=f)
        print('best overlap = ',best_overlap,file=f)

    i,j=best_overlap
    atom,label,xi,yi,zi=atom_list[i]
    atom,label,xj,yj,zj=atom_list_correct[j]
    dx,dy,dz=xj-xi,yj-yi,zj-zi
    atom_list_shifted=[]
    for atom,label,x,y,z in atom_list:
        atom_list_shifted.append((atom,label,x+dx,y+dy,z+dz))
    atom_list=atom_list_shifted[:]

    i_models=list(range(len(atom_list)))
    i_corrects=list(range(len(atom_list_correct)))
    pairs=find_pairs(i_models,i_corrects,atom_list,atom_list_correct,A)
    pairs.sort(key=lambda ss:ss[2])
    pairs.sort(key=lambda ss:ss[1])
    i_correct_previous=None
    for i_model,i_correct,d in pairs:
        if i_correct==i_correct_previous:
            pairs.remove((i_model,i_correct,d))
        i_correct_previous=i_correct

    dps=[]
    for i_model,i_correct,d in pairs:
        atom,label,x,y,z=atom_list[i_model]
        atom,label,xc,yc,zc=atom_list_correct[i_correct]
        p1=numpy.array((xc,yc,zc))
        p2=(x,y,z)
        d,p2p=d_min4(p1,p2,A)
        dp=p1-p2p
        if d<s:
            dps.append(dp)
    def target(p):
        x,y,z=p
        dr=numpy.array((-x,-y,-z))
        dt=0.0
        for dp in dps:
            dt+=dis_exact(dp+dr,A)
        return dt 
    p0=0.0,0.0,0.0
    res=minimize(target,p0,method='BFGS')
    dx,dy,dz=res.x 
    atom_list_shifted=[]
    for atom,label,x,y,z in atom_list:
        atom_list_shifted.append((atom,label,x+dx,y+dy,z+dz))
    atom_list=atom_list_shifted[:]
    save_history(atom_list,'shifted to match correct model',do_copy=True)


    i_models=list(range(len(atom_list)))
    i_corrects=list(range(len(atom_list_correct)))
    pairs=find_pairs(i_models,i_corrects,atom_list,atom_list_correct,A)
    while True:
        pairs.sort(key=lambda ss:ss[2])
        pairs.sort(key=lambda ss:ss[1])
        i_correct_previous,i_model_missed,i_correct_used=None,[],[]
        for i_model,i_correct,d in pairs:
            if i_correct==i_correct_previous:
                pairs.remove((i_model,i_correct,d))
                i_model_missed.append(i_model)
            else:
                i_correct_used.append(i_correct)
            i_correct_previous=i_correct
        if not i_model_missed: break
        i_correct_left=[]
        for i in i_corrects:
            if i not in i_correct_used:
                i_correct_left.append(i)
        pairs_missed=find_pairs(i_model_missed,i_correct_left,atom_list,atom_list_correct,A)
        pairs+=pairs_missed

    i_correct_matched=[]
    for i_model,i_correct,d in pairs:
        i_correct_matched.append(i_correct)
        atom,label,x,y,z=atom_list[i_model]
        atom,label,xc,yc,zc=atom_list_correct[i_correct]
        atom_list[i_model]=(atom,label,x,y,z)
    save_history(atom_list,'label corrected',do_copy=True)

    from datetime import datetime
    num = 0
    letters=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w',
             'x','y','z']
    mix=[]
    for i in range(10):
        for l in letters:
            mix.append(str(i)+l)
    i_a=0
    old_atom='q'
    # text = ''
    model_lines = []
    with open('history.txt','a') as f:
        f.write('\n\n\n\n\n'+str(datetime.now())+'\n')
        f.write('run number: correct marked \n\n')
        for i in range(len(atom_list_correct)):
            a,l,x,y,z=atom_list_correct[i]
            l=label_dic[a]
            if a==old_atom:
                i_a+=1  
            else:
                i_a=0
                old_atom=a  
            num += 1
            if len(a+str(num))<5:
                old_atom='q'
                str_num=str(num)
            else:
                str_num=mix[i_a]
            q = st3(a+str_num)+l
            xt = st2(round(x+0,4))
            yt = st2(round(y+0,4))
            zt = st2(round(z+0,4))
            if i in i_correct_matched:
                st = '  11.00 0.05  '
            else:
                st = '  11.00 0.10  '
            line = q+xt+yt+zt+st+'\n'
            f.write(line)
            # text += line 
            model_lines.append(line.strip())
        f.write('\n\n\n\n')
    res_lines=res_start_lines+model_lines+['','']+res_end_lines
    with open('correct_marked.res','w') as f:
        for l in res_lines:
            print(l,file=f)

    #return

    dmin,dmax=100000.0,-100000.0
    distribution=[0,0,0,0,0,0,0]
    for i_model,i_correct,d in pairs:
        if d<dmin: dmin=d 
        if d>dmax: dmax=d 
        if d<0.2: distribution[0]+=1
        if 0.2<=d<0.4: distribution[1]+=1
        if 0.4<=d<0.6: distribution[2]+=1
        if 0.6<=d<0.8: distribution[3]+=1
        if 0.8<=d<1.0: distribution[4]+=1
        if 1.0<=d<1.2: distribution[5]+=1
        if d>=1.2: distribution[6]+=1
    with open('history.txt','a') as f:
        print('0 to 0.2,',distribution[0],file=f)
        print('0.2 to 0.4,',distribution[1],file=f)
        print('0.4 to 0.6,',distribution[2],file=f)
        print('0.6 to 0.8,',distribution[3],file=f)
        print('0.8 to 1.0,',distribution[4],file=f)
        print('1.0 to 1.2,',distribution[5],file=f)
        print('1.2 to ,',distribution[6],file=f)
        print('\n\n\n')
    print('all done')
    #return

    print('minimum d = ', dmin, ',  maximum d = ', dmax,'\n\n')
    with open('history.txt','a') as f:
        print('minimum d = ', dmin, ',  maximum d = ', dmax,'\n\n',file=f)

    with open('history.txt','a') as f:
        pairs.sort(key=lambda ss:ss[2])
        count=0
        for pair in pairs:
            count+=1
            print(pair,count,file=f)

    with open('history.txt','a') as f:
        print('\n\n\n',file=f)
        pairs.sort(key=lambda ss:ss[1])
        for pair in pairs:
            print(pair,file=f)
    #return

    pairs=[]
    for i_model in range(len(atom_list)):
        atom,label,x,y,z=atom_list[i_model]
        p_model=(x,y,z)
        dmin=1000000.0
        pair=None  
        for i_correct in range(len(atom_list_correct)):
            atom,label,x,y,z=atom_list_correct[i_correct]
            p_correct=(x,y,z)
            d=d_min5(p_correct,p_model,A)
            if d<dmin:
                dmin=d 
                pair=(i_model,i_correct,dmin)
        pairs.append(pair)

    atom_list_filt=[]
    for i_model,i_correct,d in pairs:
        if d<0.8:
            atom_list_filt.append(atom_list[i_model])
    #save_history(atom_list_filt,'filtered model',do_copy=True)            

    print('\n\nall done!')

        






def nxnynz(x,y,z,Nx,Ny,Nz):
    x,y,z=put_in_cell(x),put_in_cell(y),put_in_cell(z)
    nx,ny,nz=int(x*Nx),int(y*Ny),int(z*Nz)
    return (nx,ny,nz)
def keep_in_N(nx,ny,nz,Nx,Ny,Nz):
    if nx<0: nx+=Nx  
    if ny<0: ny+=Ny  
    if nz<0: nz+=Nz  
    if nx>=Nx: nx-=Nx  
    if ny>=Ny: ny-=Ny  
    if nz>=Nz: nz-=Nz  
    return (nx,ny,nz)





def display_structure(atom_list,A):
    atomj,atom_labels,solution=atomj_solution(atom_list)
    solution=do_arrange(solution,A)

    xp,yp,zp=xpypzp(A)
    solution=to_cartesian_solution(solution,xp,yp,zp,A)

    figure=plt.figure(figsize=(9,9))
    ax2=figure.add_subplot(111,projection='3d')

    i_s=list(range(len(solution)))
    while i_s:
        i=i_s.pop()
        js=[i]
        atom=atomj[i]
        i_snew=[]
        while i_s:
            i=i_s.pop()
            if atomj[i]==atom:
                js.append(i)
            else:
                i_snew.append(i)
        xs,ys,zs=[],[],[]
        color=colors.get(atomj[js[0]],defaultcolor)
        for j in js:
            xs.append(solution[j][0])
            ys.append(solution[j][1])
            zs.append(solution[j][2])
        ax2.plot3D(xs,ys,zs,'o',color='black',markersize=6,markerfacecolor=color)
        i_s=i_snew[:]
    xs=[s[0] for s in solution]
    ys=[s[1] for s in solution]
    zs=[s[2] for s in solution]
    x1,x2=min(xs),max(xs)
    y1,y2=min(ys),max(ys)
    z1,z2=min(zs),max(zs)
    dd=max(x2-x1,y2-y1,z2-z1)
    x1,x2=(x1+x2)/2-dd/2,(x1+x2)/2+dd/2
    y1,y2=(y1+y2)/2-dd/2,(y1+y2)/2+dd/2
    z1,z2=(z1+z2)/2-dd/2,(z1+z2)/2+dd/2
    ax2.set_xlim(x1,x2)
    ax2.set_ylim(y1,y2)
    ax2.set_zlim(z1,z2)

    the_cell=[(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,0),
              (0,0,1),(1,0,1),(1,1,1),(0,1,1),(0,0,1),
              (1,0,1),(1,0,0),(1,1,0),(1,1,1),(0,1,1),(0,1,0)]
    the_cell=to_cartesian_solution(the_cell,xp,yp,zp,A)
    xs,ys,zs=[],[],[]
    for p in the_cell:
        xs.append(p[0])
        ys.append(p[1])
        zs.append(p[2])
    ax2.plot3D(xs,ys,zs,'-',color='green')

    Ns=len(solution)
    bonds=[]
    for i in range(Ns-1):
        p1=solution[i]
        r1=r_covalent.get(atomj[i],1.4)
        for j in range(i+1,Ns):
            p2=solution[j]
            r2=r_covalent.get(atomj[j],1.4)
            d=d_cartesian(p1,p2)
            if d<r1+r2+0.5:
                bonds.append((i,j))
    lines=[]
    while bonds:
        i,j=bonds.pop(0)
        line=[solution[i],solution[j]]
        new_bonds=[]
        while bonds:
            k,l=bonds.pop(0)
            if j==k:
                line.append(solution[l])
                i,j=k,l
            else:
                new_bonds.append((k,l))
        bonds=new_bonds[:]
        lines.append(line)
    for line in lines:
        xs,ys,zs=[],[],[]
        for p in line:
            xs.append(p[0])
            ys.append(p[1])
            zs.append(p[2])
        ax2.plot3D(xs,ys,zs,'-',color='blue')

    plt.show()

def animate_structure(atom_list,A):
    atomj,atom_labels,solution=atomj_solution(atom_list)
    solution=do_arrange(solution,A)

    xp,yp,zp=xpypzp(A)
    solution=to_cartesian_solution(solution,xp,yp,zp,A)

    figure=plt.figure(figsize=(9,9))
    ax2=figure.add_subplot(111,projection='3d')
    xs=[s[0] for s in solution]
    ys=[s[1] for s in solution]
    zs=[s[2] for s in solution]
    x1,x2=min(xs),max(xs)
    y1,y2=min(ys),max(ys)
    z1,z2=min(zs),max(zs)
    dd=max(x2-x1,y2-y1,z2-z1)
    x1,x2=(x1+x2)/2-dd/2,(x1+x2)/2+dd/2
    y1,y2=(y1+y2)/2-dd/2,(y1+y2)/2+dd/2
    z1,z2=(z1+z2)/2-dd/2,(z1+z2)/2+dd/2
    ax2.set_xlim(x1,x2)
    ax2.set_ylim(y1,y2)
    ax2.set_zlim(z1,z2)

    atomj_full,solution_full=atomj[:],solution[:]
    N=len(atomj)

    for i in range(1,N+1):
        atomj,solution=atomj_full[:i],solution_full[:i]
        ax2.cla()

        i_s=list(range(len(solution)))
        while i_s:
            i=i_s.pop()
            js=[i]
            atom=atomj[i]
            i_snew=[]
            while i_s:
                i=i_s.pop()
                if atomj[i]==atom:
                    js.append(i)
                else:
                    i_snew.append(i)
            xs,ys,zs=[],[],[]
            color=colors.get(atomj[js[0]],defaultcolor)
            for j in js:
                xs.append(solution[j][0])
                ys.append(solution[j][1])
                zs.append(solution[j][2])
            ax2.plot3D(xs,ys,zs,'o',color='black',markersize=6,markerfacecolor=color)
            i_s=i_snew[:]

        the_cell=[(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,0),
                  (0,0,1),(1,0,1),(1,1,1),(0,1,1),(0,0,1),
                  (1,0,1),(1,0,0),(1,1,0),(1,1,1),(0,1,1),(0,1,0)]
        the_cell=to_cartesian_solution(the_cell,xp,yp,zp,A)
        xs,ys,zs=[],[],[]
        for p in the_cell:
            xs.append(p[0])
            ys.append(p[1])
            zs.append(p[2])
        ax2.plot3D(xs,ys,zs,'-',color='green')

        Ns=len(solution)
        bonds=[]
        for i in range(Ns-1):
            p1=solution[i]
            r1=r_covalent.get(atomj[i],1.4)
            for j in range(i+1,Ns):
                p2=solution[j]
                r2=r_covalent.get(atomj[j],1.4)
                d=d_cartesian(p1,p2)
                if d<r1+r2+0.5:
                    bonds.append((i,j))
        lines=[]
        while bonds:
            i,j=bonds.pop(0)
            line=[solution[i],solution[j]]
            new_bonds=[]
            while bonds:
                k,l=bonds.pop(0)
                if j==k:
                    line.append(solution[l])
                    i,j=k,l
                else:
                    new_bonds.append((k,l))
            bonds=new_bonds[:]
            lines.append(line)
        for line in lines:
            xs,ys,zs=[],[],[]
            for p in line:
                xs.append(p[0])
                ys.append(p[1])
                zs.append(p[2])
            ax2.plot3D(xs,ys,zs,'-',color='blue')
        plt.pause(1)
    print('end of animation')
    plt.show()

def atoms_labels(molecule,Z):
    content = get_content(molecule,Z)
    return atoms_labels_from_content(content)

def atoms_labels_from_content(content):
    atoms = list(content.keys())
    atoms.sort(key=lambda a:-elements[a]['Z'])

    contents,labels = {},label_dic
    for atom in atoms:
        contents[atom]=content[atom]

    heavy_atom,heavy_label = [],[]
    for atom in atoms:
        heavy_atom += [atom]*contents[atom]
        heavy_label += [labels.setdefault(atom,'1')]*contents[atom]
    return heavy_atom,heavy_label

def get_sR1(atom_list,h,k,l,f2a,sl,Fo,Fosum,content): # calculate sR1
    (Ahj,Bhj,fcorrection,Fo,Fosum,f2a,atomj)=prep_sR1(atom_list,h,k,l,f2a,sl,Fo,Fosum,content)
    return calc_sR1(Ahj,Bhj,fcorrection,Fo,Fosum)

def prep_sR1(atom_list,h,k,l,f2a,sl,Fo,Fosum,content): # prepare for calculating sR1

    atomj,atom_labels,solution=atomj_solution(atom_list)

    remain_content={}
    for atom in content:
        remain_content[atom]=content[atom]
    for atom in atomj:
        remain_content[atom]-=1

    # calculate correction
    fcorrection=0.0*f2a[atomj[0]]
    for atom in remain_content:
        fcorrection+=f2a[atom]**2*remain_content[atom]

    fj_tmp =numpy.array([f2a[atom] for atom in atomj])
    fj=fj_tmp.T

    xj = numpy.array([s[2] for s in atom_list])
    yj = numpy.array([s[3] for s in atom_list])
    zj = numpy.array([s[4] for s in atom_list])

    angle=(6.283185306*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
        +l[:,numpy.newaxis]*zj[numpy.newaxis,:]))
    Ahj = (fj * numpy.sin(angle) )
    Bhj = (fj * numpy.cos(angle) )
    return (Ahj,Bhj,fcorrection,Fo,Fosum,f2a,atomj)

def calc_sR1(Ahj,Bhj,fcorrection,Fo,Fosum): # calculate sR1
    Ah1 =numpy.sum(Ahj ,axis=-1)
    Bh1 =numpy.sum(Bhj ,axis=-1)
    Fc = numpy.sqrt(Ah1**2+Bh1**2+fcorrection)
    r1 = (abs(Fc-Fo)).sum()/Fosum
    return r1


def identify_bad_atom(iatom,Ahj,Bhj,fcorrection,Fo,Fosum,f2a,atomj): 
    old_sR1=calc_sR1(Ahj,Bhj,fcorrection,Fo,Fosum)
    new_fcorrection=fcorrection+f2a[atomj[iatom]]**2
    new_sR1=calc_sR1(numpy.concatenate((Ahj[:,:iatom],Ahj[:,iatom+1:]),axis=1),
        numpy.concatenate((Bhj[:,:iatom],Bhj[:,iatom+1:]),axis=1),
        new_fcorrection,Fo,Fosum)
    if new_sR1<old_sR1: 
        return [iatom]
    else:
        return []


def to_sR1(atom_list,h,k,l,f2a,sl,Fo,Fosum,content): # calculating sR1
    (Ahj,Bhj,fcorrection,Fo,Fosum,f2a,atomj)=prep_sR1(atom_list,h,k,l,f2a,sl,Fo,Fosum,content)
    return calc_sR1(Ahj,Bhj,fcorrection,Fo,Fosum)



def get_structure_solution(res_file,hkl_file,molecule,Z,ntotal_initial,
    starttime,s_dual=0.4,n_refine=3,max_runs=1000,improve_only=False,
    fast=0,n_improve=1,cutlimit=1.5,extension=False,double_first=-1, 
    startfrom=1,nextra=0,mB=1,patterson=False,steps=[],cases=None):

    global NX, NY, NZ, xg, yg, zg, Z_atoms, SIN, COS, f2a, sl  
    Z_atoms = Z 

    # flag fast no longer being used

    starttime=time.time()
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('get structure solution...',tt)

    A = matrix_A(res_file)

    heavy_atom,heavy_label=atoms_labels(molecule,Z)

    light_atom = 'C'
    light_label = '1'
    Nheavy = len(heavy_atom) 
    Ntotal = Nheavy+nextra

    if ntotal_initial is None: ntotal_initial = Ntotal

    hh,kk,ll,FF2,sigFF2 = read_hkl3(hkl_file)
    sl = get_sl(hh,kk,ll,A)
    content = get_content(molecule,Z)
    scale = scaling_factor(content,sl,hh,kk,ll,FF2)
    print('scale = ', scale)
    #input('hit key...')
    FF2 *= scale 
    sigFF2 *= scale 

    # set resolution
    print('raw resolution: ', 1/2/sl.min(),1/2/sl.max())
    with open('history.txt','a') as f:
        print('\n'*4,file=f)
        print('raw resolution: ', 1/2/sl.min(),1/2/sl.max(),file=f)
    dmin=0.001 #1.0  #0.001
    print('set dmin = ', dmin)
    with open('history.txt','a') as f:
        print('set dmin = ', dmin,file=f)
    sel=sl<1/2/dmin
    #sel=(1/2<sl) * (sl<1/2/dmin)  
    #sel=sl<1/2/1.5
    #sel=sl<1/2/1.2
    #sel=sl<1/2/1.8
    h=hh[sel]
    k=kk[sel]
    l=ll[sel]
    F2=FF2[sel]
    sigF2=sigFF2[sel]

    sl = get_sl(h,k,l,A)
    # report resolution
    print('resolution used: ', 1/2/sl.min(),1/2/sl.max())
    with open('history.txt','a') as f:
        print('resolution used: ', 1/2/sl.min(),1/2/sl.max(),file=f)

    # print('writing new reflection file b.hkl: ')
    # with open('b.hkl', 'w') as f:
    #     for i in range(len(h)):
    #         f.write(st1(h[i])+st1(k[i])+st1(l[i])+st2(round(F2[i],2))+st2(round(sigF2[i],2))+'\n')
    #     f.write(st1(0)+st1(0)+st1(0)+st2(0)+st2(0)+'\n')
    # return


    # skip points
    if False:
        n=1
        ntot=int(len(h)/1)
        h=h[:ntot:n]
        k=k[:ntot:n]
        l=l[:ntot:n]
        F2=F2[:ntot:n]
        sigF2=sigF2[:ntot:n]

    # select strong peaks
    if False:
        ntotal=len(h)
        hh,kk,ll,FF2,sigFF2=[],[],[],[],[]
        for i in range(len(h)):
            if F2[i]>2*sigF2[i]: 
                hh.append(h[i])
                kk.append(k[i])
                ll.append(l[i])
                FF2.append(F2[i])
                sigFF2.append(sigF2[i])
        h,k,l,F2,sigF2=numpy.array(hh),numpy.array(kk),numpy.array(ll),numpy.array(FF2),numpy.array(sigFF2)
        print(len(h),ntotal)
        input('hit any key...')


    # set negative peaks to zero
    F2[F2<0]=0.0

    sl = get_sl(h,k,l,A)

    if True: #do Wilson scaling
        data=[]
        for i in range(len(h)):
            data.append((h[i],k[i],l[i],F2[i],sigF2[i],sl[i]))

        data.sort(key=lambda xx:xx[5])

        for i in range(len(data)):
            h[i],k[i],l[i],F2[i],sigF2[i],sl[i]=data[i]

        Nreflections=len(h)
        num_shells=5
        if Nreflections>1000: num_shells=10
        if Nreflections>10000: num_shells=20
        Nshells=int(Nreflections/num_shells)
        print(Nreflections,Nshells,num_shells)

        F2avg,ss,s2=[],[],[]
        for i in range(num_shells):
            avg=numpy.average(F2[i*Nshells:(i+1)*Nshells])
            s=numpy.average(sl[i*Nshells:(i+1)*Nshells])
            F2avg.append(avg)
            ss.append(s)
            s2.append(s*s)

        atoms,labels=atoms_labels(molecule,Z)
        lns=[]
        for i in range(len(F2avg)):
            f2sum=0.0
            for atom in atoms:
                f=fsc(ss[i],atom)
                f2sum+=f*f 
            lns.append(log(F2avg[i]/f2sum))
        s2,lns=numpy.array(s2),numpy.array(lns)

        AA=numpy.vstack([s2,numpy.ones(len(s2))]).T
        mm,cc=numpy.linalg.lstsq(AA,lns,rcond=None)[0] # slope, interception
        BB=-mm/2
        kk=exp(-cc)
        print('kk=',kk)
        print('BB=',BB)
        #BB=2.0
        #return

        F2=kk*exp(2*BB*sl*sl)*F2 

        sl = get_sl(h,k,l,A)
        content = get_content(molecule,Z)
        scale = scaling_factor(content,sl,h,k,l,F2)
        print('scale = ', scale)
        #input('hit key...')
        F2 *= scale 
        sigF2 *= scale 

        from matplotlib import pyplot
        figure=pyplot.figure(figsize=(13,7))
        ax=figure.add_subplot(111)
        line1,=ax.plot(s2,lns,'-o')
        ax.plot(s2,mm*s2+cc,'-')
        pyplot.pause(0.5)
        #pyplot.show()
        pyplot.close()

        #return

    Fo = F2.copy()
    Fo[Fo<0]=0.0
    Fo = sqrt(Fo)
    Fosum=Fo.sum()


    #sl = get_sl(h,k,l,A)

    # the whole expected molecule:
    content=get_content(molecule,Z)

    # all atomic scattering factors
    f2a = {}
    for atom in heavy_atom:
        if atom not in f2a:
            f2a[atom] = numpy.array([fsc(s,atom) for s in sl]) 

    s=0.4
    a,b,c=abc(A)
    NX, NY, NZ = int(a/s), int(b/s), int(c/s)
    xg=numpy.zeros((NX+2,NY+2,NZ+2))
    yg=numpy.zeros((NX+2,NY+2,NZ+2))
    zg=numpy.zeros((NX+2,NY+2,NZ+2))
    for ix in range(-1,NX+1):
        x=ix/NX 
        for iy in range(-1,NY+1):
            y=iy/NY
            for iz in range(-1,NZ+1):
                z=iz/NZ 
                xg[ix,iy,iz]=x 
                yg[ix,iy,iz]=y 
                zg[ix,iy,iz]=z 
    complexity=len(h)*(NX+2)*(NY+2)*(NZ+2)
    print('Nh*NX*NY*NZ = ', complexity)
    if complexity< 10: #200000000:
        try:
            hx=(h[numpy.newaxis,numpy.newaxis,numpy.newaxis,:]*xg[:,:,:,numpy.newaxis]
               +k[numpy.newaxis,numpy.newaxis,numpy.newaxis,:]*yg[:,:,:,numpy.newaxis]
               +l[numpy.newaxis,numpy.newaxis,numpy.newaxis,:]*zg[:,:,:,numpy.newaxis]) 
            COS=numpy.cos(6.283185307*hx)
            SIN=numpy.sin(6.283185307*hx)
            print(SIN.shape,SIN.size)
            print('\n\nSIN will be used\n')
        except:
            SIN, COS = None, None 
            print('\n\nSIN will not be used\n')
    else:
        print('\n\ntoo complex, SIN will not be used\n')



    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    if fast==1:
        # dual space cycling using FFT
        with open('history.txt','a') as f:
            print('\nStep : dual space cycling using FFT',tt,file=f)
            print('fast = ', fast, file=f)
            print('max_runs = ', max_runs, file=f)
            print('cutlimit = ', cutlimit, file=f)
            print('n_improve = ', n_improve, file=f)
            print('extension = ', extension, file=f)
            print('double_first = ', double_first, file=f)
            print('nextra = ', nextra, file=f)
            print('mB = ', mB, file=f)
            print('patterson = ', patterson, file=f)

        atom_list = read_atoms('a.res')
        save_history(atom_list,runs='starting model') 
        atomj,atom_labels,solution=atomj_solution(atom_list)
        nballpark=0
        r1now=to_sR1(atom_list,h,k,l,f2a,sl,Fo,Fosum,content) 
        print('0: ',Nheavy,r1now)
        with open('history.txt','a') as f:
            print('0: ',Nheavy,r1now,file=f)

        # improve solution via dual space cycles
        n = n_improve
        runs = 1

        U=0.0
        s = s_dual # 0.25   #0.4
        old_list = [] 
        base_list = atom_list[:]

        r1min=r1now
        r1min=1e200
        while True:
            tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
            atom_list = find_corrected_peaks2(h,k,l,F2,A,atom_list,heavy_atom,
                heavy_label,Nheavy,Ntotal,runs,n,U,starttime,s=s,
                n_refine=n_refine,cutlimit=cutlimit,mB=mB,patterson=patterson)
            tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
            patterson=0

            r1now=to_sR1(atom_list,h,k,l,f2a,sl,Fo,Fosum,content) 
            tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
            print(runs,r1min,r1now,tt)
            N_old=int(len(atom_list)*0.6)
            atom_list_old=atom_list[:N_old]
            if r1now>r1min: 
                runs+=1
                continue

            r1min=r1now


            save_history(atom_list,runs,do_copy=True) 
            atomj,atom_labels,solution=atomj_solution(atom_list)
            tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
            print(runs,':',r1now,tt)
            with open('history.txt','a') as f:
                print(runs,':',r1now,tt,file=f)

            if old_list:
                for j in range(len(old_list)):
                    same = True 
                    for i,(atom,label,x,y,z) in enumerate(old_list[j]):
                        try:
                            atom,label,x0,y0,z0 = atom_list[i]
                            if abs(x-x0)>0.001 or abs(y-y0)>0.001 or abs(z-z0)>0.001:
                                same=False
                                break
                        except:
                            same = False
                    if same:
                        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
                        print('converged',tt)
                        with open('history.txt','a') as f:
                            f.write('\nconverged\n'+str(tt)+'\n')
                        return
            old_list.append(atom_list[:]) 
            runs += 1
            if runs > max_runs: 
                break
            else:
                nn=int(len(atom_list)*0.6)
                atom_list=atom_list[:nn]

        return





    if fast==111:
        # sR1 in lottery mode for phasing, 2Fo-Fc recycling for output
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        with open('history.txt','a') as f:
            print('_'*80,file=f)
            print(tt,file=f)
            print('\nStep : sR1 in lottery mode-->2Fo-Fc output',file=f)
            print('fast = ', fast, file=f)
            print('max_runs = ', max_runs, file=f)
            print('cutlimit = ', cutlimit, file=f)
            print('n_improve = ', n_improve, file=f)
            print('extension = ', extension, file=f)
            print('double_first = ', double_first, file=f)
            print('nextra = ', nextra, file=f)
            print('mB = ', mB, file=f)
            print('patterson = ', patterson, file=f)
        atom_list = read_atoms('a.res')
        save_history(atom_list,runs='starting model')

        runs_dual=-1
        r1min_dual_best=1e200
        current_r1min=get_sR1(atom_list,h,k,l,f2a,sl,Fo,Fosum,content)  
        ntotal=steps[-1]
        atom_list_old,atom_list_old1,atom_list_old2=[],[],[]
        if len(atom_list)<ntotal:
            drs=[0.00001]*len(atom_list)
            n0=len(atom_list)
            for ntotal in steps:
                if len(atom_list)>=ntotal: continue
                atom_list,r1_best,solution_keep=globalmin_using_dips(h,k,l,Fo,A,molecule,
                    Z,ntotal,atom_list,starttime=starttime,runs='globalmin_using_dips')
                if r1_best<current_r1min: current_r1min=r1_best 
                drs+=solution_keep[n0:]
                n0=len(atom_list)
            atom_list_old,atom_list=atom_list[:],[]
            for i in range(len(atom_list_old)):
                if drs[i]>0: atom_list.append(atom_list_old[i])
            r1min=current_r1min 
            #save_history(atom_list_old,runs='r1min = '+str(r1min))
            #save_history(atom_list,runs='clean solution')
            tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
            with open('history.txt','a') as f:
                print(tt,file=f)
                print('_'*80,file=f)

            atom_list_dual = find_corrected_peaks2(h,k,l,F2,A,atom_list,heavy_atom,
                heavy_label,Nheavy,Ntotal,runs_dual,n_improve,U=0.0,starttime=starttime,s=s_dual,
                n_refine=n_refine,cutlimit=cutlimit,mB=mB,patterson=0)
            r1now=to_sR1(atom_list_dual,h,k,l,f2a,sl,Fo,Fosum,content) 
            runs_dual+=1 
            if r1now<r1min_dual_best:
                r1min_dual_best=r1now 
                save_history(atom_list_dual,runs='dual '+str(runs_dual)+' '+'r1 = '+str(r1now))
                tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
                with open('history.txt','a') as f:
                    print(tt,file=f)
                    print('_'*80,file=f)

        r1min=current_r1min 

        for count in range(100000):
            try:
                with open('rate.txt','r') as f:
                    text=f.read()
                    lines=text.split('\n')
                    n_rate=int(lines[0])
                    do_random_draw=int(lines[1])
                    n_random_draw=int(lines[2])
                    do_child2=int(lines[3])
            except:
                n_rate=20
                do_random_draw=0
                n_random_draw=5  
                do_child2=1 
            rate=n_rate/len(atom_list)
            if rate>0.5: rate=0.5
            if n_random_draw<1: n_random_draw=1
            print('\n\n\ncount = ',count,' rate = ',rate,' do_random_draw =',
                do_random_draw, ' n_random_draw = ', n_random_draw, ' do_child2 = ', do_child2)
            with open('history.txt','a') as f:
                print('\n\n\ncount = ',count,' rate = ',rate,' do_random_draw =',
                do_random_draw, ' n_random_draw = ', ' do_child2 = ', do_child2, n_random_draw,file=f)
            if do_random_draw:
                best_draw_r1,atom_list_best_draw=1e100,[] 
                for i_draw in range(1):
                    atom_list1=sample(atom_list,n_random_draw)
                    r11=get_sR1(atom_list1,h,k,l,f2a,sl,Fo,Fosum,content)
                    if r11<best_draw_r1:
                        best_draw_r1=r11  
                        atom_list_best_draw=atom_list1  
                        print('draw ',i_draw,best_draw_r1)
                atom_list1,atom_list2=[],[]
                for al in atom_list:
                    if al in atom_list_best_draw:
                        atom_list2.append(al)
                    else:
                        atom_list1.append(al)  
            else:           
                p0=rate*random()
                atom_list1,atom_list2=[],[]
                j_retain=-1
                j=-1
                for a_l in atom_list:
                    j+=1  
                    if j>j_retain:
                        if random()>p0:
                            atom_list1.append(a_l)
                        else:
                            atom_list2.append(a_l)
                    else:
                        atom_list1.append(a_l)
                        atom_list2.append(a_l)
            if not atom_list1: continue
            if len(atom_list1)==steps[-1]: continue
            current_r1min=1e100  
            drs=[0.00001]*len(atom_list1)
            n0=len(atom_list1)
            for ntotal in steps:
                if len(atom_list1)>=ntotal: continue
                atom_list1,r1_best,solution_keep=globalmin_using_dips(h,k,l,Fo,A,molecule,
                    Z,ntotal,atom_list1,starttime=starttime,runs='globalmin_using_dips')
                if r1_best<current_r1min: current_r1min=r1_best  
                drs+=solution_keep[n0:]
                n0=len(atom_list1)
            print('r1min = ', r1min)
            with open('history.txt','a') as f:
                print('r1min = ', r1min,file=f)
            atom_list_old1,atom_list1=atom_list1[:],[]
            for i in range(len(atom_list_old1)):
                if drs[i]>0: atom_list1.append(atom_list_old1[i])
            r1min1=current_r1min
            if r1min1<r1min:
                r1min=r1min1
                atom_list=atom_list1[:]
                print('improved model saved at count '+str(count)+' r1 = '+str(r1min))
                #save_history(atom_list_old1,runs='r1min = '+str(r1min)+' count = '+str(count))
                #save_history(atom_list1,runs='clean solution')
                tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
                with open('history.txt','a') as f:
                    print(tt,file=f)
                    print('_'*80,file=f)

                atom_list_dual = find_corrected_peaks2(h,k,l,F2,A,atom_list1,heavy_atom,
                    heavy_label,Nheavy,Ntotal,runs_dual,n_improve,U=0.0,starttime=starttime,s=s_dual,
                    n_refine=n_refine,cutlimit=cutlimit,mB=mB,patterson=0)
                r1now=to_sR1(atom_list_dual,h,k,l,f2a,sl,Fo,Fosum,content) 
                runs_dual+=1  
                if r1now<r1min_dual_best:
                    r1min_dual_best=r1now 
                    save_history(atom_list_dual,runs='dual '+str(runs_dual)+' '+'r1 = '+str(r1now))
                    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
                    print(tt)
                    with open('history.txt','a') as f:
                        print(tt,file=f)
                        print('_'*80,file=f)
            if True:
                if not do_child2: continue
                if not atom_list2:continue
                current_r1min=1e100  
                drs=[0.00001]*len(atom_list2)
                n0=len(atom_list2)
                for ntotal in steps:
                    if len(atom_list2)>=ntotal: continue
                    atom_list2,r1_best,solution_keep=globalmin_using_dips(h,k,l,Fo,A,molecule,
                        Z,ntotal,atom_list2,starttime=starttime,runs='globalmin_using_dips')
                    if r1_best<current_r1min: current_r1min=r1_best  
                    if r1_best<current_r1min: current_r1min=r1_best  
                    drs+=solution_keep[n0:]
                    n0=len(atom_list2)
                print('r1min = ', r1min)
                with open('history.txt','a') as f:
                    print('r1min = ', r1min,file=f)
                atom_list_old2,atom_list2=atom_list2[:],[]
                for i in range(len(atom_list_old2)):
                    if drs[i]>0: atom_list2.append(atom_list_old2[i])
                r1min2=current_r1min
                if r1min2<r1min:
                    r1min=r1min2
                    atom_list=atom_list2[:]
                    print('improved model saved at count '+str(count)+' r1 = '+str(r1min))
                    #save_history(atom_list_old2,runs='r1min = '+str(r1min)+' count = '+str(count))
                    #save_history(atom_list2,runs='clean solution')
                    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
                    with open('history.txt','a') as f:
                        print(tt,file=f)
                        print('_'*80,file=f)

                    atom_list_dual = find_corrected_peaks2(h,k,l,F2,A,atom_list2,heavy_atom,
                        heavy_label,Nheavy,Ntotal,runs_dual,n_improve,U=0.0,starttime=starttime,s=s_dual,
                        n_refine=n_refine,cutlimit=cutlimit,mB=mB,patterson=0)
                    r1now=to_sR1(atom_list_dual,h,k,l,f2a,sl,Fo,Fosum,content) 
                    runs_dual+=1  
                    if r1now<r1min_dual_best:
                        r1min_dual_best=r1now 
                        save_history(atom_list_dual,runs='dual '+str(runs_dual)+' '+'r1 = '+str(r1now))
                        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
                        print(tt)
                        with open('history.txt','a') as f:
                            print(tt,file=f)
                            print('_'*80,file=f)
        return








    if fast==2:
        # globalmin, using dips to narrow down candidates
        atom_list = read_atoms('a.res') 
        if len(atom_list)==1:
            atom,label,x,y,z=atom_list[0]
            x,y,z=random(),random(),random()
            atom_list[0]=(atom,label,x,y,z)
        if steps[-1] is None:
            steps=steps[:-1]
        else:
            pass #steps+=[Ntotal,]
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        with open('history.txt','a') as f:
            print('_'*80,file=f)
            print(tt,file=f)
            print('\nStep : global min using dips',file=f)
            print('fast = ', fast, file=f)
        save_history(atom_list,runs='starting model')
        if len(atom_list)>1: atom_list=relax_model(atom_list,f2a,h,k,l,Fo,Fosum,A,content,starttime)
        s_keep=[]
        for ntotal in steps:
            if len(atom_list)>=ntotal: continue
            atom_list,r1_best,solution_keep=globalmin_using_dips(h,k,l,Fo,A,molecule,
                Z,ntotal,atom_list,starttime=starttime,runs='globalmin_using_dips')
            for i in range(len(s_keep),len(solution_keep)):
                if solution_keep[i]==0.0:
                    s_keep.append(1.0)
                else:
                    s_keep.append(solution_keep[i])
        if 1:
            #atom_list=relax_model(atom_list,f2a,h,k,l,Fo,Fosum,A,content,starttime)
            idx=[]
            for i in range(len(s_keep)):
                idx.append((i,s_keep[i]))
            idx.sort(key=lambda ss:-ss[1])
            atom_list_old=atom_list[:]
            atom_list=[]
            for i,s_k in idx:
                atom_list.append(atom_list_old[i])
            r11=get_sR1(atom_list,h,k,l,f2a,sl,Fo,Fosum,content)
            save_history(atom_list,runs='final solution r1min = '+str(r11))
            tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
            with open('history.txt','a') as f:
                print('sR1 = ',r11,file=f)
                print(tt,file=f)
                print('_'*80,file=f)
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        print('Total time: ',time.time()-starttime,tt)
        with open('history.txt','a') as f:
            print('_'*80,file=f)
            print('\ntotal time: ',time.time()-starttime,'\n',file=f)
            print(tt,file=f)
            print('_'*80,file=f)
        return




    if fast==200:
        # bond length guided sR1
        atom_list = read_atoms('a.res') 
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        with open('history.txt','a') as f:
            print('_'*80,file=f)
            print(tt,file=f)
            print('\nStep : bond length guided sR1',file=f)
            print('fast = ', fast, file=f)
        save_history(atom_list,runs='starting model')
        for case in cases:
            ntotal=len(atom_list)+1
            atom_list,r1_best,solution_keep=globalmin_using_dips(h,k,l,Fo,A,molecule,
                Z,ntotal,atom_list,starttime=starttime,
                runs='globalmin_using_dips',more_info=case)
        if 1:
            r11=get_sR1(atom_list,h,k,l,f2a,sl,Fo,Fosum,content)
            save_history(atom_list,runs='bond length guided sR1 r1min = '+str(r11))
            tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
            with open('history.txt','a') as f:
                print('sR1 = ',r11,file=f)
                print(tt,file=f)
                print('_'*80,file=f)
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        print('Total time: ',time.time()-starttime,tt)
        with open('history.txt','a') as f:
            print('_'*80,file=f)
            print('\ntotal time: ',time.time()-starttime,'\n',file=f)
            print(tt,file=f)
            print('_'*80,file=f)
        return




    if fast==222:
        # sR1 in lottery mode
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        with open('history.txt','a') as f:
            print('_'*80,file=f)
            print(tt,file=f)
            print('\nStep : sR1 in lottery mode',file=f)
            print('fast = ', fast, file=f)
        atom_list = read_atoms('a.res')
        if len(atom_list)==1:
            atom,label,x,y,z=atom_list[0]
            x,y,z=random(),random(),random()
            atom_list[0]=(atom,label,x,y,z)
        if len(atom_list)>1: atom_list=relax_model(atom_list,f2a,h,k,l,Fo,Fosum,A,content,starttime)
        save_history(atom_list,runs='starting model')

        current_r1min=get_sR1(atom_list,h,k,l,f2a,sl,Fo,Fosum,content)  
        ntotal=steps[-1]
        atom_list_old,atom_list_old1,atom_list_old2=atom_list[:],[],[]
        if len(atom_list)<ntotal:
            drs=[0.00001]*len(atom_list)
            n0=len(atom_list)
            for ntotal in steps:
                if len(atom_list)>=ntotal: continue
                atom_list,r1_best,solution_keep=globalmin_using_dips(h,k,l,Fo,A,molecule,
                    Z,ntotal,atom_list,starttime=starttime,runs='globalmin_using_dips')
                if r1_best<current_r1min: current_r1min=r1_best 
                drs+=solution_keep[n0:]
                n0=len(atom_list)
            atom_list_old,atom_list=atom_list[:],[]
            for i in range(len(atom_list_old)):
                if drs[i]>0: atom_list.append(atom_list_old[i])
        r1min=get_sR1(atom_list_old,h,k,l,f2a,sl,Fo,Fosum,content) 
        if 1:
            save_history(atom_list_old,runs='full solution r1min = '+str(r1min))
            atom_list=atom_list_old[:]
            atom_list_new=relax_model(atom_list,f2a,h,k,l,Fo,Fosum,A,content,starttime)
            r1min_new=get_sR1(atom_list_new,h,k,l,f2a,sl,Fo,Fosum,content)
            if r1min_new<r1min: 
                r1min=r1min_new
                atom_list=atom_list_new[:]
                save_history(atom_list,runs='clean solution r1min = '+str(r1min_new))
            tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
            with open('history.txt','a') as f:
                print(tt,file=f)
                print('_'*80,file=f)

        for count in range(100000):
            try:
                with open('rate.txt','r') as f:
                    text=f.read()
                    lines=text.split('\n')
                    n_rate=int(lines[0])
                    do_random_draw=int(lines[1])
                    n_random_draw=int(lines[2])
                    do_child2=int(lines[3])
            except:
                n_rate=20
                do_random_draw=0
                n_random_draw=5  
                do_child2=1 
            rate=n_rate/len(atom_list)
            if rate>0.5: rate=0.5
            if n_random_draw<1: n_random_draw=1
            print('\n\n\ncount = ',count,' rate = ',rate,' do_random_draw =',
                do_random_draw, ' n_random_draw = ', n_random_draw, ' do_child2 = ', do_child2)
            with open('history.txt','a') as f:
                print('\n\n\ncount = ',count,' rate = ',rate,' do_random_draw =',
                do_random_draw, ' n_random_draw = ', ' do_child2 = ', do_child2, n_random_draw,file=f)
            if do_random_draw:
                best_draw_r1,atom_list_best_draw=1e100,[] 
                for i_draw in range(1):
                    atom_list1=sample(atom_list,n_random_draw)
                    r11=get_sR1(atom_list1,h,k,l,f2a,sl,Fo,Fosum,content)
                    if r11<best_draw_r1:
                        best_draw_r1=r11  
                        atom_list_best_draw=atom_list1  
                        print('draw ',i_draw,best_draw_r1)
                atom_list1,atom_list2=[],[]
                for al in atom_list:
                    if al in atom_list_best_draw:
                        atom_list2.append(al)
                    else:
                        atom_list1.append(al)  
            else:           
                p0=rate*random()
                atom_list1,atom_list2=[],[]
                j_retain=-1
                j=-1
                for a_l in atom_list:
                    j+=1  
                    if j>j_retain:
                        if random()>p0:
                            atom_list1.append(a_l)
                        else:
                            atom_list2.append(a_l)
                    else:
                        atom_list1.append(a_l)
                        atom_list2.append(a_l)
            if not atom_list1: continue
            if len(atom_list1)==steps[-1]: continue
            atom_list1=relax_model(atom_list1,f2a,h,k,l,Fo,Fosum,A,content,starttime)
            current_r1min=1e100  
            drs=[0.00001]*len(atom_list1)
            n0=len(atom_list1)
            for ntotal in steps:
                if len(atom_list1)>=ntotal: continue
                atom_list1,r1_best,solution_keep=globalmin_using_dips(h,k,l,Fo,A,molecule,
                    Z,ntotal,atom_list1,starttime=starttime,runs='globalmin_using_dips')
                if r1_best<current_r1min: current_r1min=r1_best  
                drs+=solution_keep[n0:]
                n0=len(atom_list1)
            print('r1min = ', r1min)
            with open('history.txt','a') as f:
                print('r1min = ', r1min,file=f)
            r1min1=get_sR1(atom_list1,h,k,l,f2a,sl,Fo,Fosum,content)
            if r1min1<r1min:
                r1min=r1min1
                atom_list=atom_list1[:]
                print('improved model saved at count '+str(count)+' r1 = '+str(r1min))
                save_history(atom_list,runs='full solution r1min = '+str(r1min)+' count = '+str(count))
                tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
                with open('history.txt','a') as f:
                    print(tt,file=f)
                    print('_'*80,file=f)
            atom_list_old1,atom_list1=atom_list1[:],[]
            for i in range(len(atom_list_old1)):
                if drs[i]>0: atom_list1.append(atom_list_old1[i])
            atom_list1=relax_model(atom_list1,f2a,h,k,l,Fo,Fosum,A,content,starttime)
            r1min1=get_sR1(atom_list1,h,k,l,f2a,sl,Fo,Fosum,content)
            if r1min1<r1min:
                r1min=r1min1
                atom_list=atom_list1[:]
                print('improved model saved at count '+str(count)+' r1 = '+str(r1min))
                save_history(atom_list,runs='clean solution r1min = '+str(r1min)+' count = '+str(count))
                tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
                with open('history.txt','a') as f:
                    print(tt,file=f)
                    print('_'*80,file=f)
            else:
                if not do_child2: continue
                if not atom_list2:continue
                current_r1min=1e100  
                drs=[0.00001]*len(atom_list2)
                n0=len(atom_list2)
                for ntotal in steps:
                    if len(atom_list2)>=ntotal: continue
                    atom_list2,r1_best,solution_keep=globalmin_using_dips(h,k,l,Fo,A,molecule,
                        Z,ntotal,atom_list2,starttime=starttime,runs='globalmin_using_dips')
                    if r1_best<current_r1min: current_r1min=r1_best  
                    if r1_best<current_r1min: current_r1min=r1_best  
                    drs+=solution_keep[n0:]
                    n0=len(atom_list2)
                print('r1min = ', r1min)
                with open('history.txt','a') as f:
                    print('r1min = ', r1min,file=f)
                r1min2=get_sR1(atom_list2,h,k,l,f2a,sl,Fo,Fosum,content)
                if r1min2<r1min:
                    r1min=r1min2
                    atom_list=atom_list2[:]
                    n_fail=0
                    r1min_fail_best=1e100 
                    print('improved model saved at count '+str(count)+' r1 = '+str(r1min))
                    save_history(atom_list,runs='full solution r1min = '+str(r1min)+' count = '+str(count))
                    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
                    with open('history.txt','a') as f:
                        print(tt,file=f)
                        print('_'*80,file=f)
                atom_list_old2,atom_list2=atom_list2[:],[]
                for i in range(len(atom_list_old2)):
                    if drs[i]>0: atom_list2.append(atom_list_old2[i])
                atom_list2=relax_model(atom_list2,f2a,h,k,l,Fo,Fosum,A,content,starttime)
                r1min2=get_sR1(atom_list2,h,k,l,f2a,sl,Fo,Fosum,content)
                if r1min2<r1min:
                    r1min=r1min2
                    atom_list=atom_list2[:]
                    print('improved model saved at count '+str(count)+' r1 = '+str(r1min))
                    save_history(atom_list,runs='clean solution r1min = '+str(r1min)+' count = '+str(count))
                    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
                    with open('history.txt','a') as f:
                        print(tt,file=f)
                        print('_'*80,file=f)
        return





    if fast==2220:
        # sR1 in lottery mode, always try both child models
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        with open('history.txt','a') as f:
            print('_'*80,file=f)
            print(tt,file=f)
            print('\nStep : sR1 in lottery mode',file=f)
            print('fast = ', fast, file=f)
        atom_list = read_atoms('a.res')
        if len(atom_list)==1:
            atom,label,x,y,z=atom_list[0]
            x,y,z=random(),random(),random()
            atom_list[0]=(atom,label,x,y,z)
        if len(atom_list)>1: atom_list=relax_model(atom_list,f2a,h,k,l,Fo,Fosum,A,content,starttime)
        save_history(atom_list,runs='starting model')

        current_r1min=get_sR1(atom_list,h,k,l,f2a,sl,Fo,Fosum,content)  
        ntotal=steps[-1]
        atom_list_old,atom_list_old1,atom_list_old2=atom_list[:],[],[]
        if len(atom_list)<ntotal:
            drs=[0.00001]*len(atom_list)
            n0=len(atom_list)
            for ntotal in steps:
                if len(atom_list)>=ntotal: continue
                atom_list,r1_best,solution_keep=globalmin_using_dips(h,k,l,Fo,A,molecule,
                    Z,ntotal,atom_list,starttime=starttime,runs='globalmin_using_dips')
                if r1_best<current_r1min: current_r1min=r1_best 
                drs+=solution_keep[n0:]
                n0=len(atom_list)
            atom_list_old,atom_list=atom_list[:],[]
            for i in range(len(atom_list_old)):
                if drs[i]>0: atom_list.append(atom_list_old[i])
        r1min=get_sR1(atom_list_old,h,k,l,f2a,sl,Fo,Fosum,content) 
        if 1:
            save_history(atom_list_old,runs='full solution r1min = '+str(r1min))
            atom_list=atom_list_old[:]
            atom_list_new=relax_model(atom_list,f2a,h,k,l,Fo,Fosum,A,content,starttime)
            r1min_new=get_sR1(atom_list_new,h,k,l,f2a,sl,Fo,Fosum,content)
            if r1min_new<r1min: 
                r1min=r1min_new
                atom_list=atom_list_new[:]
                save_history(atom_list,runs='clean solution r1min = '+str(r1min_new))
            tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
            with open('history.txt','a') as f:
                print(tt,file=f)
                print('_'*80,file=f)

        for count in range(100000):
            try:
                with open('rate.txt','r') as f:
                    text=f.read()
                    lines=text.split('\n')
                    n_rate=int(lines[0])
                    do_random_draw=int(lines[1])
                    n_random_draw=int(lines[2])
                    do_child2=int(lines[3])
            except:
                n_rate=20
                do_random_draw=0
                n_random_draw=5  
                do_child2=1 
            rate=n_rate/len(atom_list)
            if rate>0.5: rate=0.5
            if n_random_draw<1: n_random_draw=1
            print('\n\n\ncount = ',count,' rate = ',rate,' do_random_draw =',
                do_random_draw, ' n_random_draw = ', n_random_draw, ' do_child2 = ', do_child2)
            with open('history.txt','a') as f:
                print('\n\n\ncount = ',count,' rate = ',rate,' do_random_draw =',
                do_random_draw, ' n_random_draw = ', ' do_child2 = ', do_child2, n_random_draw,file=f)
            if do_random_draw:
                best_draw_r1,atom_list_best_draw=1e100,[] 
                for i_draw in range(1):
                    atom_list1=sample(atom_list,n_random_draw)
                    r11=get_sR1(atom_list1,h,k,l,f2a,sl,Fo,Fosum,content)
                    if r11<best_draw_r1:
                        best_draw_r1=r11  
                        atom_list_best_draw=atom_list1  
                        print('draw ',i_draw,best_draw_r1)
                atom_list1,atom_list2=[],[]
                for al in atom_list:
                    if al in atom_list_best_draw:
                        atom_list2.append(al)
                    else:
                        atom_list1.append(al)  
            else:           
                p0=rate*random()
                atom_list1,atom_list2=[],[]
                j_retain=-1
                j=-1
                for a_l in atom_list:
                    j+=1  
                    if j>j_retain:
                        if random()>p0:
                            atom_list1.append(a_l)
                        else:
                            atom_list2.append(a_l)
                    else:
                        atom_list1.append(a_l)
                        atom_list2.append(a_l)
            if not atom_list1: continue
            if len(atom_list1)==steps[-1]: continue
            atom_list1=relax_model(atom_list1,f2a,h,k,l,Fo,Fosum,A,content,starttime)
            current_r1min=1e100  
            drs=[0.00001]*len(atom_list1)
            n0=len(atom_list1)
            for ntotal in steps:
                if len(atom_list1)>=ntotal: continue
                atom_list1,r1_best,solution_keep=globalmin_using_dips(h,k,l,Fo,A,molecule,
                    Z,ntotal,atom_list1,starttime=starttime,runs='globalmin_using_dips')
                if r1_best<current_r1min: current_r1min=r1_best  
                drs+=solution_keep[n0:]
                n0=len(atom_list1)
            print('r1min = ', r1min)
            with open('history.txt','a') as f:
                print('r1min = ', r1min,file=f)
            r1min1=get_sR1(atom_list1,h,k,l,f2a,sl,Fo,Fosum,content)
            if r1min1<r1min:
                r1min=r1min1
                atom_list=atom_list1[:]
                print('improved model saved at count '+str(count)+' r1 = '+str(r1min))
                save_history(atom_list,runs='full solution r1min = '+str(r1min)+' count = '+str(count))
                tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
                with open('history.txt','a') as f:
                    print(tt,file=f)
                    print('_'*80,file=f)
            atom_list_old1,atom_list1=atom_list1[:],[]
            for i in range(len(atom_list_old1)):
                if drs[i]>0: atom_list1.append(atom_list_old1[i])
            atom_list1=relax_model(atom_list1,f2a,h,k,l,Fo,Fosum,A,content,starttime)
            r1min1=get_sR1(atom_list1,h,k,l,f2a,sl,Fo,Fosum,content)
            if r1min1<r1min:
                r1min=r1min1
                atom_list=atom_list1[:]
                print('improved model saved at count '+str(count)+' r1 = '+str(r1min))
                save_history(atom_list,runs='clean solution r1min = '+str(r1min)+' count = '+str(count))
                tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
                with open('history.txt','a') as f:
                    print(tt,file=f)
                    print('_'*80,file=f)
            if 1:
                if not do_child2: continue
                if not atom_list2:continue
                current_r1min=1e100  
                drs=[0.00001]*len(atom_list2)
                n0=len(atom_list2)
                for ntotal in steps:
                    if len(atom_list2)>=ntotal: continue
                    atom_list2,r1_best,solution_keep=globalmin_using_dips(h,k,l,Fo,A,molecule,
                        Z,ntotal,atom_list2,starttime=starttime,runs='globalmin_using_dips')
                    if r1_best<current_r1min: current_r1min=r1_best  
                    if r1_best<current_r1min: current_r1min=r1_best  
                    drs+=solution_keep[n0:]
                    n0=len(atom_list2)
                print('r1min = ', r1min)
                with open('history.txt','a') as f:
                    print('r1min = ', r1min,file=f)
                r1min2=get_sR1(atom_list2,h,k,l,f2a,sl,Fo,Fosum,content)
                if r1min2<r1min:
                    r1min=r1min2
                    atom_list=atom_list2[:]
                    n_fail=0
                    r1min_fail_best=1e100 
                    print('improved model saved at count '+str(count)+' r1 = '+str(r1min))
                    save_history(atom_list,runs='full solution r1min = '+str(r1min)+' count = '+str(count))
                    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
                    with open('history.txt','a') as f:
                        print(tt,file=f)
                        print('_'*80,file=f)
                atom_list_old2,atom_list2=atom_list2[:],[]
                for i in range(len(atom_list_old2)):
                    if drs[i]>0: atom_list2.append(atom_list_old2[i])
                atom_list2=relax_model(atom_list2,f2a,h,k,l,Fo,Fosum,A,content,starttime)
                r1min2=get_sR1(atom_list2,h,k,l,f2a,sl,Fo,Fosum,content)
                if r1min2<r1min:
                    r1min=r1min2
                    atom_list=atom_list2[:]
                    print('improved model saved at count '+str(count)+' r1 = '+str(r1min))
                    save_history(atom_list,runs='clean solution r1min = '+str(r1min)+' count = '+str(count))
                    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
                    with open('history.txt','a') as f:
                        print(tt,file=f)
                        print('_'*80,file=f)
        return






    if fast==2221:
        # systematically searching initial single atom position
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        with open('history.txt','a') as f:
            print('_'*80,file=f)
            print(tt,file=f)
            print('\nStep : study starting with single atom',file=f)
            print('fast = ', fast, file=f)
        atom_list = read_atoms('a.res')
        if 0:
            atom,label,x,y,z=atom_list[0]
            a,b,c=abc(A)
            Nx,Ny,Nz=int(a/0.4),int(b/0.4),int(c/0.4)
            x,y,z=int(0.3*Nx)/Nx,int(0.3*Ny)/Ny,int(0.3*Nz)/Nz  
            atom_list=[(atom,label,x,y,z)]
        save_history(atom_list,runs='starting model')

        r1min=get_sR1(atom_list,h,k,l,f2a,sl,Fo,Fosum,content)
        count_best=0  

        a,b,c=abc(A)
        ss=0.4
        Nx,Ny,Nz=int(a/ss),int(b/ss),int(c/ss)
        atom,label,x0,y0,z0=atom_list[0]
        Ncut=4
        atom_lists=[]
        for ix in range(Ncut):
            for iy in range(Ncut):
                for iz in range(Ncut):
                    x,y,z=x0+ix/Ncut/Nx,y0+iy/Ncut/Ny,z0+iz/Ncut/Nz
                    atom_lists.append([(atom,label,x,y,z)])
        NTOT=Ncut*Ncut*Ncut 

        count=0
        for atom_list1 in atom_lists:
            count+=1
            print('\n\ncount = ',count,'/',NTOT)
            with open('history.txt','a') as f:
                print('\n\ncount = ',count,'/',NTOT,file=f)
            current_r1min=1e100  
            drs=[0.00001]*len(atom_list1)
            n0=len(atom_list1)
            for ntotal in steps:
                if len(atom_list1)>=ntotal: continue
                atom_list1,r1_best,solution_keep=globalmin_using_dips(h,k,l,Fo,A,molecule,
                    Z,ntotal,atom_list1,starttime=starttime,runs='globalmin_using_dips')
                if r1_best<current_r1min: current_r1min=r1_best  
                drs+=solution_keep[n0:]
                n0=len(atom_list1)
            atom_list_old1,atom_list1=atom_list1[:],[]
            for i in range(len(atom_list_old1)):
                if drs[i]>0: atom_list1.append(atom_list_old1[i])
            r1min1=current_r1min
            print(atom_list_old1[0])
            print('r1min1 = ',r1min1)
            print('\n')
            with open('history.txt','a') as f:
                print(atom_list_old1[0],file=f)
                print('r1min1 = ',r1min1,file=f)
                print('\n',file=f)
            if r1min1<r1min:
                r1min=r1min1
                count_best=count  
                atom_list=atom_list1[:]
                print('improved model saved at count '+str(count)+' r1min = '+str(r1min))
                save_history(atom_list_old1,runs='r1min = '+str(r1min)+' count = '+str(count))
                save_history(atom_list1,runs='cleaned solution')
                tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
                with open('history.txt','a') as f:
                    print(tt,file=f)
                    print('_'*80,file=f)
            print('count_best = ',count_best,'r1min = ',r1min)
            print('\n')
            with open('history.txt','a') as f:
                print('count_best = ',count_best,'r1min = ',r1min,file=f)
                print('\n',file=f)
        return




    if fast==2223:
        # sR1 in systematic lottory mode, with pre-filtering of starting partial models
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        with open('history.txt','a') as f:
            print('_'*80,file=f)
            print(tt,file=f)
            print('\nStep : sR1 in systematic lottery mode',file=f)
            print('fast = ', fast, file=f)
        atom_list = read_atoms('a.res')
        if len(atom_list)==1:
            atom,label,x,y,z=atom_list[0]
            x,y,z=random(),random(),random()
            atom_list[0]=(atom,label,x,y,z)
        if len(atom_list)>1: atom_list=relax_model(atom_list,f2a,h,k,l,Fo,Fosum,A,content,starttime)
        save_history(atom_list,runs='starting model')

        current_r1min=get_sR1(atom_list,h,k,l,f2a,sl,Fo,Fosum,content)  
        ntotal=steps[-1]
        atom_list_old,atom_list_old1,atom_list_old2=atom_list[:],[],[]
        if len(atom_list)<ntotal:
            drs=[0.00001]*len(atom_list)
            n0=len(atom_list)
            for ntotal in steps:
                if len(atom_list)>=ntotal: continue
                atom_list,r1_best,solution_keep=globalmin_using_dips(h,k,l,Fo,A,molecule,
                    Z,ntotal,atom_list,starttime=starttime,runs='globalmin_using_dips')
                if r1_best<current_r1min: current_r1min=r1_best 
                drs+=solution_keep[n0:]
                n0=len(atom_list)
            atom_list_old,atom_list=atom_list[:],[]
            for i in range(len(atom_list_old)):
                if drs[i]>0: atom_list.append(atom_list_old[i])
        r1min=get_sR1(atom_list_old,h,k,l,f2a,sl,Fo,Fosum,content) 
        if 1:
            save_history(atom_list_old,runs='full solution r1min = '+str(r1min))
            atom_list=atom_list_old[:]
            atom_list_new=relax_model(atom_list,f2a,h,k,l,Fo,Fosum,A,content,starttime)
            r1min_new=get_sR1(atom_list_new,h,k,l,f2a,sl,Fo,Fosum,content)
            if r1min_new<r1min: 
                r1min=r1min_new
                atom_list=atom_list_new[:]
                save_history(atom_list,runs='clean solution r1min = '+str(r1min_new))
            tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
            with open('history.txt','a') as f:
                print(tt,file=f)
                print('_'*80,file=f)

        for count in range(100000):
            try:
                with open('rate.txt','r') as f:
                    text=f.read()
                    lines=text.split('\n')
                    n_rate=int(lines[0])
                    do_random_draw=int(lines[1])
                    n_random_draw=int(lines[2])
                    do_child2=int(lines[3])
            except:
                n_rate=20
                do_random_draw=0
                n_random_draw=5  
                do_child2=1 
            n_atom_list=len(atom_list)
            n_random_draw=1+int(20*random())
            j_retain=0
            if not do_random_draw:
                if random()<0.5:
                    n_select=n_random_draw
                else:
                    n_select=n_atom_list-n_random_draw-j_retain
            else:
                n_select=n_atom_list-n_random_draw
            if n_select > 20:
                n_trials=200
            else:
                n_trials=1000
            if n_select>=n_atom_list-j_retain: n_select=n_atom_list-j_retain
            if n_select<=0: n_select=1

            print('\ncount = ',count,'n_select = ', n_select,'n_trials = ',n_trials,
                'r1min = ',r1min,round(time.time()-starttime,1),'\n')
            with open('history.txt','a') as f:
                print('\n\n\ncount = ',count,'n_select = ', n_select,'n_trials = ',n_trials,
                'r1min = ',r1min,round(time.time()-starttime,1),'\n',file=f)
            draw_start=time.time()
            improved=False
            best_draw_r1,atom_list_best_draw=1e100,[] 
            for i_draw in range(n_trials):
            #while (time.time()-starttime<120):
                atom_list1=atom_list[:j_retain]+sample(atom_list[j_retain:],n_select)
                r11=get_sR1(atom_list1,h,k,l,f2a,sl,Fo,Fosum,content)
                if r11<best_draw_r1:
                    improved=True
                    best_draw_r1=r11  
                    atom_list_best_draw=atom_list1  
                    print('draw ',i_draw,best_draw_r1,round(time.time()-draw_start,1),
                        round(time.time()-starttime,1))
            if not improved: 
                #print('no improvement in starting partial model...')
                continue
            atom_list2=atom_list_best_draw[:]
            atom_list2=relax_model(atom_list2,f2a,h,k,l,Fo,Fosum,A,content,starttime)
            current_r1min=1e100  
            drs=[0.00001]*len(atom_list2)
            n0=len(atom_list2)
            for ntotal in steps:
                if len(atom_list2)>=ntotal: continue
                atom_list2,r1_best,solution_keep=globalmin_using_dips(h,k,l,Fo,A,molecule,
                    Z,ntotal,atom_list2,starttime=starttime,runs='globalmin_using_dips')
                if r1_best<current_r1min: current_r1min=r1_best  
                drs+=solution_keep[n0:]
                n0=len(atom_list2)
            print('r1min = ', r1min)
            with open('history.txt','a') as f:
                print('r1min = ', r1min,file=f)
            r1min2=get_sR1(atom_list2,h,k,l,f2a,sl,Fo,Fosum,content)
            if r1min2<r1min:
                atom_list=atom_list2[:]
                r1min=r1min2
                print('\nimproved model saved at count '+str(count)+' r1 = '+str(r1min)+'\n\n\n')
                save_history(atom_list2,runs='full solution r1min = '+str(r1min)+' count = '+str(count))
                tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
                with open('history.txt','a') as f:
                    print(tt,file=f)
                    print('_'*80,file=f)
                best_draw_r1,atom_list_best_draw=1e100,[]
            atom_list_old2,atom_list2=atom_list2[:],[]
            for i in range(len(atom_list_old2)):
                if drs[i]>0: atom_list2.append(atom_list_old2[i])
            atom_list2=relax_model(atom_list2,f2a,h,k,l,Fo,Fosum,A,content,starttime)
            r1min2=get_sR1(atom_list2,h,k,l,f2a,sl,Fo,Fosum,content)
            if r1min2<r1min:
                atom_list=atom_list2[:]
                r1min=r1min2
                print('\nimproved model saved at count '+str(count)+' r1 = '+str(r1min)+'\n\n\n')
                save_history(atom_list2,runs='clean solution r1min = '+str(r1min)+' count = '+str(count))
                tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
                with open('history.txt','a') as f:
                    print(tt,file=f)
                    print('_'*80,file=f)
                best_draw_r1,atom_list_best_draw=1e100,[]

        return






    if fast==2224:
        # searching starting partial model
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        with open('history.txt','a') as f:
            print('_'*80,file=f)
            print(tt,file=f)
            print('\nStep : searching starting partial model',file=f)
            print('fast = ', fast, file=f)
        atom_list = read_atoms('a.res')
        if len(atom_list)>1: atom_list=relax_model(atom_list,f2a,h,k,l,Fo,Fosum,A,content,starttime)
        save_history(atom_list,runs='starting model')

        heavy_atom,heavy_label=atoms_labels(molecule,Z)
        the_N=12
        the_atoms,the_labels=heavy_atom[:the_N],heavy_label[:the_N]

        if 0:
            ntotal=steps[-1]
            peaks=peaks_for_globalmin_using_dips(h,k,l,Fo,A,molecule,Z,ntotal,atom_list,
                starttime=starttime,runs='peaks for globalmin_using_dips')
        the_peaks=[]
        for atom,label,x,y,z in atom_list:
            the_peaks.append((x,y,z))
        if 0:
            for x,y,z,ff in peaks[:int(len(heavy_atom)/2)]:
                the_peaks.append((x,y,z))

        r1min=1e200 

        for count in range(100000):
            tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
            print('\ncount = ',count,
                tt,'\n')
            with open('history.txt','a') as f:
                print('\n\n\ncount = ',count,
                    tt,'\n',file=f)
            draw_start=time.time()
            best_draw_r1,atom_list_best_draw=1e100,[] 
            to_continue=1
            i_draw=0
            while (to_continue and time.time()-draw_start<120):
                i_draw+=1
                selected_peaks=sample(the_peaks,the_N)
                atom_list1=to_atom_list(the_atoms,the_labels,selected_peaks)
                r11=get_sR1(atom_list1,h,k,l,f2a,sl,Fo,Fosum,content)
                if r11<best_draw_r1:
                    best_draw_r1=r11  
                    atom_list_best_draw=atom_list1  
                    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
                    print('draw ',i_draw,best_draw_r1,round(time.time()-draw_start,1),
                        tt)
                    with open('history.txt','a') as f:
                        print('draw ',i_draw,best_draw_r1,round(time.time()-draw_start,1),
                            tt,file=f)
                try:
                    with open('rate.txt','r') as f:
                        text=f.read()
                        lines=text.split('\n')
                        to_continue=int(lines[4])
                except:
                    to_continue=1
            atom_list2=atom_list_best_draw[:]
            atom_list2=relax_model(atom_list2,f2a,h,k,l,Fo,Fosum,A,content,starttime)
            current_r1min=1e100  
            drs=[0.00001]*len(atom_list2)
            n0=len(atom_list2)
            for ntotal in steps:
                if len(atom_list2)>=ntotal: continue
                atom_list2,r1_best,solution_keep=globalmin_using_dips(h,k,l,Fo,A,molecule,
                    Z,ntotal,atom_list2,starttime=starttime,runs='globalmin_using_dips')
                if r1_best<current_r1min: current_r1min=r1_best  
                drs+=solution_keep[n0:]
                n0=len(atom_list2)
            print('r1min = ', r1min)
            with open('history.txt','a') as f:
                print('r1min = ', r1min,file=f)
            r1min2=get_sR1(atom_list2,h,k,l,f2a,sl,Fo,Fosum,content)
            if r1min2<r1min:
                atom_list=atom_list2[:]
                r1min=r1min2
                print('\nimproved model saved at count '+str(count)+' r1 = '+str(r1min)+'\n\n\n')
                save_history(atom_list2,runs='full solution r1min = '+str(r1min)+' count = '+str(count))
                tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
                with open('history.txt','a') as f:
                    print(tt,file=f)
                    print('_'*80,file=f)
            atom_list_old2,atom_list2=atom_list2[:],[]
            for i in range(len(atom_list_old2)):
                if drs[i]>0: atom_list2.append(atom_list_old2[i])
            atom_list2=relax_model(atom_list2,f2a,h,k,l,Fo,Fosum,A,content,starttime)
            r1min2=get_sR1(atom_list2,h,k,l,f2a,sl,Fo,Fosum,content)
            if r1min2<r1min:
                atom_list=atom_list2[:]
                r1min=r1min2
                print('\nimproved model saved at count '+str(count)+' r1 = '+str(r1min)+'\n\n\n')
                save_history(atom_list2,runs='clean solution r1min = '+str(r1min)+' count = '+str(count))
                tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
                with open('history.txt','a') as f:
                    print(tt,file=f)
                    print('_'*80,file=f)

        return






    if fast==2225:
        # randomly searching starting partial model
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        with open('history.txt','a') as f:
            print('_'*80,file=f)
            print(tt,file=f)
            print('\nStep : randomly searching starting partial model',file=f)
            print('fast = ', fast, file=f)

        atom_list = read_atoms('a.res')
        if len(atom_list)>1: atom_list=relax_model(atom_list,f2a,h,k,l,Fo,Fosum,A,content,starttime)
        save_history(atom_list,runs='starting model')

        current_r1min=get_sR1(atom_list,h,k,l,f2a,sl,Fo,Fosum,content)  
        ntotal=steps[-1]
        atom_list_old,atom_list_old1,atom_list_old2=atom_list[:],[],[]
        if len(atom_list)<ntotal:
            drs=[0.00001]*len(atom_list)
            n0=len(atom_list)
            for ntotal in steps:
                if len(atom_list)>=ntotal: continue
                atom_list,r1_best,solution_keep=globalmin_using_dips(h,k,l,Fo,A,molecule,
                    Z,ntotal,atom_list,starttime=starttime,runs='globalmin_using_dips')
                if r1_best<current_r1min: current_r1min=r1_best 
                drs+=solution_keep[n0:]
                n0=len(atom_list)
            atom_list_old,atom_list=atom_list[:],[]
            for i in range(len(atom_list_old)):
                if drs[i]>0: atom_list.append(atom_list_old[i])
        r1min=get_sR1(atom_list_old,h,k,l,f2a,sl,Fo,Fosum,content) 
        if 1:
            save_history(atom_list_old,runs='full solution r1min = '+str(r1min))
            atom_list=atom_list_old[:]
            atom_list_new=relax_model(atom_list,f2a,h,k,l,Fo,Fosum,A,content,starttime)
            r1min_new=get_sR1(atom_list_new,h,k,l,f2a,sl,Fo,Fosum,content)
            if r1min_new<r1min: 
                r1min=r1min_new
                atom_list=atom_list_new[:]
                save_history(atom_list,runs='clean solution r1min = '+str(r1min_new))
            tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
            with open('history.txt','a') as f:
                print(tt,file=f)
                print('_'*80,file=f)

        heavy_atom,heavy_label=atoms_labels(molecule,Z)
        the_N=230
        the_atoms,the_labels=heavy_atom[:the_N],heavy_label[:the_N]

        for count in range(100000):
            tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
            print('\ncount = ',count,
                round(time.time()-starttime,1),tt,'\n')
            with open('history.txt','a') as f:
                print('\n\n\ncount = ',count,
                    round(time.time()-starttime,1),tt,'\n',file=f)
            draw_start=time.time()
            best_draw_r1,atom_list_best_draw=1e100,[] 
            to_continue=1
            i_draw=0
            while (to_continue and time.time()-draw_start<600):
                i_draw+=1
                atom_list1=sample(atom_list,the_N)
                r11=get_sR1(atom_list1,h,k,l,f2a,sl,Fo,Fosum,content)
                if r11<best_draw_r1:
                    best_draw_r1=r11  
                    atom_list_best_draw=atom_list1  
                    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
                    print('draw ',i_draw,best_draw_r1,round(time.time()-draw_start,1),
                        tt)
                    with open('history.txt','a') as f:
                        print('draw ',i_draw,best_draw_r1,round(time.time()-draw_start,1),
                            tt,file=f)
                try:
                    with open('rate.txt','r') as f:
                        text=f.read()
                        lines=text.split('\n')
                        to_continue=int(lines[4])
                except:
                    to_continue=1
                #to_continue=0
            atom_list2=atom_list_best_draw[:]
            atom_list2=relax_model(atom_list2,f2a,h,k,l,Fo,Fosum,A,content,starttime)
            current_r1min=1e100  
            drs=[0.00001]*len(atom_list2)
            n0=len(atom_list2)
            for ntotal in steps:
                if len(atom_list2)>=ntotal: continue
                atom_list2,r1_best,solution_keep=globalmin_using_dips(h,k,l,Fo,A,molecule,
                    Z,ntotal,atom_list2,starttime=starttime,runs='globalmin_using_dips')
                if r1_best<current_r1min: current_r1min=r1_best  
                drs+=solution_keep[n0:]
                n0=len(atom_list2)
            print('r1min = ', r1min)
            with open('history.txt','a') as f:
                print('r1min = ', r1min,file=f)
            r1min2=get_sR1(atom_list2,h,k,l,f2a,sl,Fo,Fosum,content)
            if r1min2<r1min:
                atom_list=atom_list2[:]
                r1min=r1min2
                print('\nimproved model saved at count '+str(count)+' r1 = '+str(r1min)+'\n\n\n')
                save_history(atom_list2,runs='full solution r1min = '+str(r1min)+' count = '+str(count))
                tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
                with open('history.txt','a') as f:
                    print(tt,file=f)
                    print('_'*80,file=f)
            atom_list_old2,atom_list2=atom_list2[:],[]
            for i in range(len(atom_list_old2)):
                if drs[i]>0: atom_list2.append(atom_list_old2[i])
            atom_list2=relax_model(atom_list2,f2a,h,k,l,Fo,Fosum,A,content,starttime)
            r1min2=get_sR1(atom_list2,h,k,l,f2a,sl,Fo,Fosum,content)
            if r1min2<r1min:
                atom_list=atom_list2[:]
                r1min=r1min2
                print('\nimproved model saved at count '+str(count)+' r1 = '+str(r1min)+'\n\n\n')
                save_history(atom_list2,runs='clean solution r1min = '+str(r1min)+' count = '+str(count))
                tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
                with open('history.txt','a') as f:
                    print(tt,file=f)
                    print('_'*80,file=f)
            else:
                continue
                atom_list=atom_list2[:]

        return








    fragment0,n_fold=make_benzene()
    #fragment0,n_fold=make_benzenestar()
    #fragment0,n_fold=make_ethynylbenzene()
    #fragment0,n_fold=make_PF6()
    #fragment0,n_fold=make_pentagon()
    #fragment0,n_fold=make_molecule('molecule_C60.txt')
    #fragment0,n_fold=make_molecule('molecule_C7.txt')
    #fragment0,n_fold=make_invert_molecule()
    #fragment0=make_linear('S2.txt')
    #fragment0=make_S2()

    if fast==3:
        # find fragment orientation 
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        with open('history.txt','a') as f:
            print('\nStep : find fragment orientations',file=f)
            print(tt,file=f)
            print('fast = ', fast, file=f)
        if 1:
            atom_list0=[]
            p0=(0.3,0.3,0.3)
        else:
            atom_list0 = read_atoms('a.res')
            j=7
            atom,label,x,y,z=atom_list0[j-1]
            p0=(x,y,z) 
        find_fragment_orientations(h,k,l,Fo,A,molecule,Z,atom_list0,fragment0,n_fold,p0,
            starttime=starttime,runs='find fragment orientations')
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        print('Total time: ',time.time()-starttime,tt)
        atom_list1=read_atoms('a.res')
        r11=get_sR1(atom_list1,h,k,l,f2a,sl,Fo,Fosum,content)
        print('r1 = ',r11)
        with open('history.txt','a') as f:
            print('r1 = ',r11,file=f)
            print('_'*80,file=f)
            print('\ntotal time: ',time.time()-starttime,'\n',file=f)
            print(tt,file=f)
            print('_'*80,file=f)

    if fast==4:
        # filt orientations 
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        with open('history.txt','a') as f:
            print('\nStep : locate batch of fragments',file=f)
            print(tt,file=f)
            print('fast = ', fast, file=f)
        atom_list=[]
        filt_orientations(h,k,l,Fo,A,molecule,Z,fragment0,atom_list,
            s=0.4,starttime=starttime,runs='find fragment locations')
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        print('Total time: ',time.time()-starttime,tt)
        with open('history.txt','a') as f:
            print('_'*80,file=f)
            print('\ntotal time: ',time.time()-starttime,'\n',file=f)
            print(tt,file=f)
            print('_'*80,file=f)



    if fast==30:
        # find linear fragment orientation 
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        with open('history.txt','a') as f:
            print('\nStep : find linear orientations',file=f)
            print(tt,file=f)
            print('fast = ', fast, file=f)
        find_linear_orientations(h,k,l,Fo,A,molecule,Z,fragment0,
            starttime=starttime,runs='find linear orientations')
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        print('Total time: ',time.time()-starttime,tt)
        with open('history.txt','a') as f:
            print('_'*80,file=f)
            print('\ntotal time: ',time.time()-starttime,'\n',file=f)
            print(tt,file=f)
            print('_'*80,file=f)



    if fast==5:
        # locate fragments 
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        with open('history.txt','a') as f:
            print('\nStep : locate batch of fragments',file=f)
            print(tt,file=f)
            print('fast = ', fast, file=f)
        atom_list=[]
        locate_a_batch_of_fragments(h,k,l,Fo,A,molecule,Z,fragment0,
            starttime=starttime,runs='locate a batch of fragments')
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        print('Total time: ',time.time()-starttime,tt)
        with open('history.txt','a') as f:
            print('_'*80,file=f)
            print('\ntotal time: ',time.time()-starttime,'\n',file=f)
            print(tt,file=f)
            print('_'*80,file=f)





    if fast==50:
        # locate linear fragments 
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        with open('history.txt','a') as f:
            print('\nStep : locate batch of linear fragments',file=f)
            print(tt,file=f)
            print('fast = ', fast, file=f)
        atom_list=[]
        locate_a_batch_of_linear_fragments(h,k,l,Fo,A,molecule,Z,fragment0,
            starttime=starttime,runs='locate a batch of linear fragments')
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        print('Total time: ',time.time()-starttime,tt)
        with open('history.txt','a') as f:
            print('_'*80,file=f)
            print('\ntotal time: ',time.time()-starttime,'\n',file=f)
            print(tt,file=f)
            print('_'*80,file=f)





    if fast==51:
        # find P4 locations 
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        with open('history.txt','a') as f:
            print('\nStep : find P4 locations',file=f)
            print(tt,file=f)
            print('fast = ', fast, file=f)
        atom_list=[]
        find_P4_locations(h,k,l,Fo,A,molecule,Z,fragment0,
            starttime=starttime,runs='find P4 locations')
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        print('Total time: ',time.time()-starttime,tt)
        with open('history.txt','a') as f:
            print('_'*80,file=f)
            print('\ntotal time: ',time.time()-starttime,'\n',file=f)
            print(tt,file=f)
            print('_'*80,file=f)








    if fast==52:
        # placing a fragment 
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        with open('history.txt','a') as f:
            print('\nStep : placing a fragment',file=f)
            print(tt,file=f)
            print('fast = ', fast, file=f)
        fragment0_file='frag_S2O2C12.txt'
        atom_list=read_atoms('a.res')
        placing_fragment(h,k,l,Fo,A,molecule,Z,fragment0_file,atom_list,
            starttime=starttime,runs='placing a fragment')
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        print('Total time: ',time.time()-starttime,tt)
        with open('history.txt','a') as f:
            print('_'*80,file=f)
            print('\ntotal time: ',time.time()-starttime,'\n',file=f)
            print(tt,file=f)
            print('_'*80,file=f)



    if fast==53:
        # pattern recognition
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        with open('history.txt','a') as f:
            print('\nStep : pattern recognition',file=f)
            print(tt,file=f)
            print('fast = ', fast, file=f)
        atom_list=read_atoms('a.res')

        ss=0.8
        fragments=[]
        fragments.append(make_C3_sp2())
        # fragments.append(make_C3_sp3())
        # fragments.append(make_CNC())
        # fragments.append(make_CCON())
        # fragments.append(make_SS())
        # fragments.append(make_SC())
        # fragments.append(make_benzene_tip())

        a,b,c=abc(A)
        Nx,Ny,Nz=int(a/ss),int(b/ss),int(c/ss)
        def get_clean_model(atom_list,ss,fragment0,maxita):
            original_model=set()
            for i in range(len(atom_list)):
                atom,label,x,y,z=atom_list[i]
                nx,ny,nz=nxnynz(x,y,z,Nx,Ny,Nz)
                if (nx,ny,nz) in original_model:
                    print('atom ', i, ' is redundent')
                    with open('history.txt','a') as f:
                        print('atom ',i,' is redundent',file=f)
                original_model.add((nx,ny,nz))
            original_model_list=list(original_model)

            s=5.0
            maxpsi,maxphi=360.0,180.0
            na,nb,nc = int(maxpsi/s),int(maxphi/s),int(maxita/s)
            X = numpy.array([i*maxpsi/na for i in range(na)])
            Y = numpy.array([i*maxphi/nb for i in range(nb)])
            Z = numpy.array([i*maxita/nc for i in range(nc)])

            p0=(0.3,0.3,0.3)
            C,D=getCD(A)
            clean_model=set()
            for psi in X:
                for phi in Y:
                    for ita in Z:
                        fragment=add_fragment(p0,psi,phi,ita,D,fragment0)
                        atom,label,x,y,z=fragment[0]
                        nx0,ny0,nz0=nxnynz(x,y,z,Nx,Ny,Nz)
                        fragment_model=set()
                        for i in range(1,len(fragment)):
                            atom,label,x,y,z=fragment[i]
                            nx,ny,nz=nxnynz(x,y,z,Nx,Ny,Nz)
                            nx,ny,nz=nx-nx0,ny-ny0,nz-nz0
                            fragment_model.add((nx,ny,nz))
                        fragment_model_list=list(fragment_model)
                        for (nx,ny,nz) in original_model_list:
                            matched=True 
                            for (dnx,dny,dnz) in fragment_model_list:
                                fnx,fny,fnz=nx+dnx,ny+dny,nz+dnz
                                fnx,fny,fnz=keep_in_N(fnx,fny,fnz,Nx,Ny,Nz)
                                if (fnx,fny,fnz) not in original_model:
                                    matched=False
                                    break
                            if matched:
                                clean_model.add((nx,ny,nz))
                                for (dnx,dny,dnz) in fragment_model_list:
                                    clean_model.add((nx+dnx,ny+dny,nz+dnz))
                                    for (dnx,dny,dnz) in fragment_model_list:
                                        fnx,fny,fnz=nx+dnx,ny+dny,nz+dnz
                                        fnx,fny,fnz=keep_in_N(fnx,fny,fnz,Nx,Ny,Nz)
                                        clean_model.add((fnx,fny,fnz))
            return clean_model

        clean_model=set()
        i=0
        for fragment0,n_fold in fragments:
            i+=1
            print('start pattern ',i)
            maxita=360/n_fold
            clean_model=clean_model.union(get_clean_model(atom_list,ss,fragment0,maxita))

        final_model=set()
        final_atom_list=[]
        for atom,label,x,y,z in atom_list:
            nx,ny,nz=nxnynz(x,y,z,Nx,Ny,Nz)
            if (nx,ny,nz) in clean_model:
                if (nx,ny,nz) not in final_model:
                    final_model.add((nx,ny,nz))
                    final_atom_list.append((atom,label,x,y,z))

        atomj,atom_labels,solution=atomj_solution(final_atom_list)
        for i in range(len(solution)):
            x,y,z=solution[i]
            x,y,z=put_in_cell(x),put_in_cell(y),put_in_cell(z)
            solution[i]=(x,y,z)
        solution=do_arrange(solution,A)
        final_atom_list = to_atom_list(atomj,atom_labels,solution)

        save_history(final_atom_list,'pattern recognition',True)
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        print('Total time: ',time.time()-starttime,tt)
        with open('history.txt','a') as f:
            print('_'*80,file=f)
            print('\ntotal time: ',time.time()-starttime,'\n',file=f)
            print(tt,file=f)
            print('_'*80,file=f)





    if fast==6:
        # relax a model 
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        with open('history.txt','a') as f:
            print('\nStep : relax a model',file=f)
            print(tt,file=f)
            print('fast = ', fast, file=f)
        atom_list=read_atoms('a.res')
        atom_list=relax_model(atom_list,f2a,h,k,l,Fo,Fosum,A,content,starttime)
        save_history(atom_list,runs='relax model')

        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        print('Total time: ',time.time()-starttime,tt)
        with open('history.txt','a') as f:
            print('_'*80,file=f)
            print('\ntotal time: ',time.time()-starttime,'\n',file=f)
            print(tt,file=f)
            print('_'*80,file=f)






    if fast==7:
        # tweaking a model 
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        with open('history.txt','a') as f:
            print('\nStep : relax a model',file=f)
            print(tt,file=f)
            print('fast = ', fast, file=f)
        atom_list=read_atoms('a.res')

        atomj,atom_labels,solution=atomj_solution(atom_list)

        remain_content={}
        for atom in content:
            remain_content[atom]=content[atom]
        for atom in atomj:
            remain_content[atom]-=1

        # calculate correction
        fcorrection=0.0*f2a[atomj[0]]
        for atom in remain_content:
            fcorrection+=f2a[atom]**2*remain_content[atom]

        fj_tmp =numpy.array([f2a[atom] for atom in atomj])
        fj=fj_tmp.T

        xj = numpy.array([s[2] for s in atom_list])
        yj = numpy.array([s[3] for s in atom_list])
        zj = numpy.array([s[4] for s in atom_list])

        r1min=gen_sR1(h,k,l,xj,yj,zj,fj,fcorrection,Fo,Fosum)

        a,b,c=abc(A)
        dx,dy,dz=0.2/a,0.2/b,0.2/c  
        n=len(xj)
        for num in range(1000000):
            print(num)
            improved=False
            for i in range(10):
                xjnew,yjnew,zjnew=(xj+dx*(numpy.random.rand(n)-0.5),
                    yj+dy*(numpy.random.rand(n)-0.5),
                    zj+dz*(numpy.random.rand()-0.5))
                r1=gen_sR1(h,k,l,xjnew,yjnew,zjnew,fj,fcorrection,Fo,Fosum)
                if r1<r1min:
                    improved=True 
                    xj,yj,zj=xjnew,yjnew,zjnew
            for i in range(len(atom_list)):
                atom,label,x,y,z=atom_list[i]
                atom_list[i]=atom,label,xj[i],yj[i],zj[i]

            if improved:
                tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
                print('improved num = ',num,'r1min = ',r1min,'Total time: ',time.time()-starttime,tt)
                save_history(atom_list,runs='tweaking model num = '+str(num))

                with open('history.txt','a') as f:
                    print('_'*80,file=f)
                    print('\ntotal time: ',time.time()-starttime,'\n',file=f)
                    print(tt,file=f)
                    print('_'*80,file=f)



    if fast==8:
        # shake a model 
        tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
        with open('history.txt','a') as f:
            print('\nStep : shake a model',file=f)
            print(tt,file=f)
            print('fast = ', fast, file=f)
        atom_list=read_atoms('a.res')

        atomj,atom_labels,solution=atomj_solution(atom_list)

        remain_content={}
        for atom in content:
            remain_content[atom]=content[atom]
        for atom in atomj:
            remain_content[atom]-=1

        # calculate correction
        fcorrection=0.0*f2a[atomj[0]]
        for atom in remain_content:
            fcorrection+=f2a[atom]**2*remain_content[atom]

        fj_tmp =numpy.array([f2a[atom] for atom in atomj])
        fj=fj_tmp.T

        xj = numpy.array([s[2] for s in atom_list])
        yj = numpy.array([s[3] for s in atom_list])
        zj = numpy.array([s[4] for s in atom_list])

        r1min=gen_sR1(h,k,l,xj,yj,zj,fj,fcorrection,Fo,Fosum)
        print('starting r1min = ',r1min)
        with open('history.txt','a') as f:
            print('\nstarting r1min = ',r1min,'\n',file=f)

        n_atoms,n_trials=10,2000
        with open('history.txt','a') as f:
            print('n_atoms, n_trials = ',n_atoms,n_trials,file=f)

        (atom_list,r1min,improved)=shake_model(atom_list,h,k,l,f2a,sl,Fo,Fosum,content,A,n_atoms,n_trials,starttime)

        if improved:
            tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
            print('improved r1min = ',r1min,'Total time: ',time.time()-starttime,tt)
            save_history(atom_list,runs='shake model improved r1min = '+str(r1min))

            with open('history.txt','a') as f:
                print('_'*80,file=f)
                print('\ntotal time: ',time.time()-starttime,'\n',file=f)
                print(tt,file=f)
                print('_'*80,file=f)




def relax_model(atom_list,f2a,h,k,l,Fo,Fosum,A,content,starttime):
    #return atom_list
    atomj,atom_labels,solution=atomj_solution(atom_list)
    remain_content={}
    for atom in content:
        remain_content[atom]=content[atom]
    for atom in atomj:
        remain_content[atom]-=1
    # calculate correction
    fcorrection=0.0*f2a[atomj[0]]
    for atom in remain_content:
        fcorrection+=f2a[atom]**2*remain_content[atom]

    fj_tmp =numpy.array([f2a[atom] for atom in atomj])
    fj=fj_tmp.T

    xj = numpy.array([s[2] for s in atom_list])
    yj = numpy.array([s[3] for s in atom_list])
    zj = numpy.array([s[4] for s in atom_list])

    chj=(6.283185306*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
        +l[:,numpy.newaxis]*zj[numpy.newaxis,:]))
    Ahj = (fj * numpy.sin(chj) )
    Bhj = (fj * numpy.cos(chj) )

    Ah1 =numpy.sum(Ahj ,axis=-1)
    Bh1 =numpy.sum(Bhj ,axis=-1)

    Fc = numpy.sqrt(Ah1**2+Bh1**2+fcorrection)
    r1start = (abs(Fc-Fo)).sum()/Fosum
    tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
    print('start r1 = ',r1start,tt)

    a,b,c=abc(A)
    s=0.4

    ixs=list(range(len(atom_list)))
    improved=True
    while improved:
        improved=False
        ixs.sort(key=lambda x:random())
        for i in ixs:
            atom,label,x,y,z=atom_list[i]
            f2=f2a[atom]
            cold=6.283185306*(h*x+k*y+l*z)
            Ah1-=f2*numpy.sin(cold)
            Bh1-=f2*numpy.cos(cold)
            sx,sy,sz=2*s/a,2*s/b,2*s/c 
            r1min,xbest,ybest,zbest=quick_sR1(x,y,z,f2,Ah1,Bh1,Fo,Fosum,fcorrection,h,k,l),x,y,z 
            for repeat in range(3):
                x,y,z=xbest,ybest,zbest  
                sx,sy,sz=sx/2,sy/2,sz/2
                for ix in range(-1,2):
                    xnew=x+ix*sx 
                    for iy in range(-1,2):
                        ynew=y+iy*sy 
                        for iz in range(-1,2):
                            znew=z+iz*sz 
                            r1=quick_sR1(xnew,ynew,znew,f2,Ah1,Bh1,Fo,Fosum,fcorrection,h,k,l)
                            if r1<r1min:
                                improved=True 
                                r1min,xbest,ybest,zbest=r1,xnew,ynew,znew
            atom_list[i]=(atom,label,xbest,ybest,zbest)
            cnew=6.283185306*(h*xbest+k*ybest+l*zbest)
            Ah1+=(f2*numpy.sin(cnew))
            Bh1+=(f2*numpy.cos(cnew))
        if improved:
            tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
            print('improved, r1 = ',r1min,tt)
            with open('history.txt','a') as f:
                print('improved, r1 = ',r1min,tt,file=f)




    return atom_list 

def quick_sR1(xnew,ynew,znew,f2,Ah1,Bh1,Fo,Fosum,fcorrection,h,k,l): # calculate sR1
    cnew=6.283185306*(h*xnew+k*ynew+l*znew)
    Ah1n=Ah1+(f2*numpy.sin(cnew))
    Bh1n=Bh1+(f2*numpy.cos(cnew))
    Fc = numpy.sqrt(Ah1n**2+Bh1n**2+fcorrection)
    r1 = (abs(Fc-Fo)).sum()/Fosum
    return r1


def tweaking_model(atom_list,h,k,l,f2a,sl,Fo,Fosum,content,A,starttime):
    atomj,atom_labels,solution=atomj_solution(atom_list)

    remain_content={}
    for atom in content:
        remain_content[atom]=content[atom]
    for atom in atomj:
        remain_content[atom]-=1

    # calculate correction
    fcorrection=0.0*f2a[atomj[0]]
    for atom in remain_content:
        fcorrection+=f2a[atom]**2*remain_content[atom]

    fj_tmp =numpy.array([f2a[atom] for atom in atomj])
    fj=fj_tmp.T

    xj = numpy.array([s[2] for s in atom_list])
    yj = numpy.array([s[3] for s in atom_list])
    zj = numpy.array([s[4] for s in atom_list])

    r1min=gen_sR1(h,k,l,xj,yj,zj,fj,fcorrection,Fo,Fosum)

    a,b,c=abc(A)
    dx,dy,dz=2/a,2/b,2/c  
    n=len(xj)
    improved=False
    for i in range(1000):
        xjnew,yjnew,zjnew=xj+dx*numpy.random.rand(n),yj+dy*numpy.random.rand(n),zj+dz*numpy.random.rand()
        r1=gen_sR1(h,k,l,xjnew,yjnew,zjnew,fj,fcorrection,Fo,Fosum)
        if r1<r1min:
            improved=True 
            xj,yj,zj=xjnew,yjnew,zjnew
    for i in range(len(atom_list)):
        atom,label,x,y,z=atom_list[i]
        atom_list[i]=atom,label,xj[i],yj[i],zj[i]

    return (atom_list,r1min,improved)

def gen_sR1(h,k,l,xj,yj,zj,fj,fcorrection,Fo,Fosum): # calculate sR1

    angle=(6.283185306*(h[:,numpy.newaxis]*xj[numpy.newaxis,:]+k[:,numpy.newaxis]*yj[numpy.newaxis,:]
        +l[:,numpy.newaxis]*zj[numpy.newaxis,:]))
    Ahj = (fj * numpy.sin(angle) )
    Bhj = (fj * numpy.cos(angle) )

    Ah1 =numpy.sum(Ahj ,axis=-1)
    Bh1 =numpy.sum(Bhj ,axis=-1)
    Fc = numpy.sqrt(Ah1**2+Bh1**2+fcorrection)
    r1 = (abs(Fc-Fo)).sum()/Fosum
    return r1



def shake_model(atom_list,h,k,l,f2a,sl,Fo,Fosum,content,A,n_atoms,n_trials,starttime):
    atomj,atom_labels,solution=atomj_solution(atom_list)

    remain_content={}
    for atom in content:
        remain_content[atom]=content[atom]
    for atom in atomj:
        remain_content[atom]-=1

    # calculate correction
    fcorrection=0.0*f2a[atomj[0]]
    for atom in remain_content:
        fcorrection+=f2a[atom]**2*remain_content[atom]

    fj_tmp =numpy.array([f2a[atom] for atom in atomj])
    fj=fj_tmp.T

    xj = numpy.array([s[2] for s in atom_list])
    yj = numpy.array([s[3] for s in atom_list])
    zj = numpy.array([s[4] for s in atom_list])

    r1min=gen_sR1(h,k,l,xj,yj,zj,fj,fcorrection,Fo,Fosum)

    a,b,c=abc(A)
    dx,dy,dz=0.02/a,0.02/b,0.02/c  
    n=len(xj)
    js=list(range(n))
    improved=False
    ngood=0
    for i in range(n_trials):
        c0=numpy.zeros(n)
        js_select=sample(js,n_atoms)
        for j in js_select: c0[j]=1.0
        dxj,dyj,dzj=c0*dx*(numpy.random.rand(n)-0.5),c0*dy*(numpy.random.rand(n)-0.5),c0*dz*(numpy.random.rand()-0.5)
        xjnew,yjnew,zjnew=xj+dxj,yj+dyj,zj+dzj
        r1=gen_sR1(h,k,l,xjnew,yjnew,zjnew,fj,fcorrection,Fo,Fosum)
        print(i,r1,r1min)
        if r1<r1min-1e-8:
            improved=True 
            xj,yj,zj=xjnew,yjnew,zjnew
            r1min=r1 
            ngood+=1 
            print('good ',ngood,r1)
            with open('history.txt','a') as f:
                print('good ',ngood,' out of ',i,r1,file=f)
            better=True 
            while better:
                xjnew,yjnew,zjnew=xj+0.1*dxj,yj+0.1*dyj,zj+0.1*dzj
                r1=gen_sR1(h,k,l,xjnew,yjnew,zjnew,fj,fcorrection,Fo,Fosum)
                if r1<r1min-1e-8:
                    xj,yj,zj=xjnew,yjnew,zjnew
                    r1min=r1 
                    print('better',r1)
                    with open('history.txt','a') as f:
                        print('better',r1,file=f)
                else:
                    better=False
            better=True 
            while better:
                xjnew,yjnew,zjnew=xj-0.1*dxj,yj-0.1*dyj,zj-0.1*dzj
                r1=gen_sR1(h,k,l,xjnew,yjnew,zjnew,fj,fcorrection,Fo,Fosum)
                if r1<r1min-1e-8:
                    xj,yj,zj=xjnew,yjnew,zjnew
                    r1min=r1 
                    print('better',r1)
                    with open('history.txt','a') as f:
                        print('better',r1,file=f)
                else:
                    better=False
            better=True 
            while better:
                xjnew,yjnew,zjnew=xj+0.01*dxj,yj+0.01*dyj,zj+0.01*dzj
                r1=gen_sR1(h,k,l,xjnew,yjnew,zjnew,fj,fcorrection,Fo,Fosum)
                if r1<r1min-1e-8:
                    xj,yj,zj=xjnew,yjnew,zjnew
                    r1min=r1 
                    print('better',r1)
                    with open('history.txt','a') as f:
                        print('better',r1,file=f)
                else:
                    better=False
            better=True 
            while better:
                xjnew,yjnew,zjnew=xj-0.01*dxj,yj-0.01*dyj,zj-0.01*dzj
                r1=gen_sR1(h,k,l,xjnew,yjnew,zjnew,fj,fcorrection,Fo,Fosum)
                if r1<r1min-1e-8:
                    xj,yj,zj=xjnew,yjnew,zjnew
                    r1min=r1 
                    print('better',r1)
                    with open('history.txt','a') as f:
                        print('better',r1,file=f)
                else:
                    better=False
            better=True 
            while better:
                xjnew,yjnew,zjnew=xj+0.001*dxj,yj+0.001*dyj,zj+0.001*dzj
                r1=gen_sR1(h,k,l,xjnew,yjnew,zjnew,fj,fcorrection,Fo,Fosum)
                if r1<r1min-1e-8:
                    xj,yj,zj=xjnew,yjnew,zjnew
                    r1min=r1 
                    print('better',r1)
                    with open('history.txt','a') as f:
                        print('better',r1,file=f)
                else:
                    better=False
            better=True 
            while better:
                xjnew,yjnew,zjnew=xj-0.001*dxj,yj-0.001*dyj,zj-0.001*dzj
                r1=gen_sR1(h,k,l,xjnew,yjnew,zjnew,fj,fcorrection,Fo,Fosum)
                if r1<r1min-1e-8:
                    xj,yj,zj=xjnew,yjnew,zjnew
                    r1min=r1 
                    print('better',r1)
                    with open('history.txt','a') as f:
                        print('better',r1,file=f)
                else:
                    better=False
            #break
    for i in range(len(atom_list)):
        atom,label,x,y,z=atom_list[i]
        atom_list[i]=atom,label,xj[i],yj[i],zj[i]

    return (atom_list,r1min,improved)




