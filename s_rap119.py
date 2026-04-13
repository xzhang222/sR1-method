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

print('START calculation...',tt)
res_file='a.res'
hkl_file='a.hkl'
A = matrix_A(res_file)

molecule = 'C78' 
Z = 2
fast=2
          #1: dual space cycling using FFT
          #2: one cycle of sR1 method
          #222: sR1 method in lottery mode
          #3: find fragment orientations
          #4: filt orientations
          #5: locate fragments
steps=[10,30,80,156]
cases=[('C','1',153,1.39,0.3),]
startfrom=1 #r1 hole,min-cycles, gambling: atom to start from                          
max_runs=1000000  #max dual-space or min cycles
cutlimit = 4.0  #dual-space: 1.5 probably enough, may choose 2.5
nextra = 0
mB = 1 # defaul 1, how much to sharpen F2, 0 no sharpen, 2 fully sharpened, 1 halfway sharpened
patterson = 0 # 0 start res, 1 patterson, 2 random phase angle             

ntotal_initial= None # num atoms in min, if none will decide automatically 
s_dual=0.35
n_refine=3 # num of cycles of refinement
n_improve = 1 # 1 for 2*Fo-1*Fc
improve_only = 1 # this flag no longer being used
extension=False   # recommend False
double_first=-1  # -1 not to double, 1 to double first run for catching weaker peaks                

if Z<0:
    Z = 1
    print('When Z=1, density = ',round(density(molecule,Z,A),2))
    while True:
        Z = int(input('Please enter Z: '))
        print('When Z=',Z,', density = ',round(density(molecule,Z,A),2))
        done = input('Done? ')
        if done: sys.exit()
print('When Z=',Z,', density = ',round(density(molecule,Z,A),2))

get_structure_solution(res_file,hkl_file,molecule,Z,ntotal_initial,
    starttime,s_dual,n_refine,max_runs,improve_only,fast=fast,
    n_improve=n_improve,cutlimit=cutlimit,extension=extension,
    double_first=double_first,startfrom=startfrom,nextra=nextra,
    mB=mB,patterson=patterson,steps=steps,cases=cases)

with open('history.txt','a') as f:
    print('\n\n\nstep:  \n\n\n',file=f)
print('done')
