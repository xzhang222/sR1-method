import sys
import time 
from tools import *

with open(sys.argv[0],'r') as f:
    text=f.read()
with open('history.txt','a') as f:
    print('\n\nprogram '+sys.argv[0]+':\n'+text+'\n\n',file=f)

# show r1 from a.res and a.hkl
if 0:
    get_r1(0)
    
if 0:
    re_order()   

if 0:
    generate_random_model()   

if 0:
    atom_list = read_atoms('a.res')
    save_history(atom_list,runs='current model')   

if 0:
    re_arrange() 

if 1:
    pairing(correct_res='correct.res',res='a.res')

if 0:
    compare_models2(correct_res='correct.res',init_res='a.res',
        compare_txt='compare.txt',Ntry=None,r0=0.500)
    
with open('history.txt','a') as f:
    print('\n\n\nstep:  \n\n\n',file=f)
print('done')

