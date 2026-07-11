import sys
import time 
from tools import *

molecule = 'C78'
Z = 2
fast=2
          #1: 2Fo-Fc dual space recycling
          #2: sR1 #2223: sR1 in lottery mode #200: bond length guided sR1
          #3: find fragment orientations  #4: filter orientations
          #5: locate fragments
# for sR1
steps=[10,30,80,156]
# for bond length guided sR1
cases=[('C',9,1.39,0.3),('C',152,1.39,0.3),('C',153,1.39,0.3),
       ('C',154,1.39,0.3),('C',155,1.39,0.3),]
# for pR1 
s_angle=20.0 
max_orientations=10
free_standing=1  # 1: very first 0: add to a partial structure
orientation_selected=[0,1] # orientations are labeled as 0, 1, 2...
orientation_file='orientations_benzenestar.txt'

#fragment0,n_fold=make_molecule('C5.txt')
#fragment0,n_fold=make_invert_molecule('C7O.txt')
#fragment0,n_fold=make_fragment_from_mol2('dan3')
#fragment0,n_fold=make_C6O3()
#fragment0,n_fold=make_PdCl4()
#fragment0,n_fold=make_benzene()
fragment0,n_fold=make_benzenestar()
#fragment0,n_fold=make_ethynylbenzene()
#fragment0,n_fold=make_PF6()
#fragment0,n_fold=make_pentagon()
#fragment0,n_fold=make_molecule('molecule_C60.txt')
#fragment0,n_fold=make_molecule('molecule_C7.txt')
#fragment0,n_fold=make_invert_molecule()
#fragment0=make_linear('S2.txt')
#fragment0=make_S2()

starttime=time.time()
tt='('+time.strftime('%Y-%m-%d, %H:%M:%S',time.localtime())+') '+str(round(time.time()-starttime,1))
with open(sys.argv[0],'r') as f:
    text=f.read()
with open('history.txt','a') as f:
    print('\n\nstep: \n',tt,file=f)
    print('\n\nprogram '+sys.argv[0]+':\n'+text+'\n\n',file=f)
if 1:
    with open('tools.py','r') as f:
        text=f.read()
    with open('history.txt','a') as f:
        print('\n\nprogram '+'tools.py'+':\n'+text+'\n\n',file=f)

print('START calculation...',tt)
res_file='a.res'
hkl_file='a.hkl'
A = matrix_A(res_file)

if Z<0:
    Z = 1
    print('When Z=1, density = ',round(density(molecule,Z,A),2))
    while True:
        Z = int(input('Please enter Z: '))
        print('When Z=',Z,', density = ',round(density(molecule,Z,A),2))
        done = input('Done? ')
        if done: sys.exit()
print('When Z=',Z,', density = ',round(density(molecule,Z,A),2))

get_structure_solution(res_file,hkl_file,molecule,Z,starttime,fast=fast,
    steps=steps,cases=cases,free_standing=free_standing,
    orientation_selected=orientation_selected,fragment0=fragment0,
    n_fold=n_fold,orientation_file=orientation_file,
    max_orientations=max_orientations,s_angle=s_angle)

with open('history.txt','a') as f:
    print('\n\n\nstep:  \n\n\n',file=f)
print('done')
