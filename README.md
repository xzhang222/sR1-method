# sR1-method
Python codes for single-atom R1 (sR1) and partial-structure R1 (pR1) calculations

About single-atom R1 (sR1) calculation:

The included a.hkl and a.res provide one example of raw data files. At the start, a.res provides initial model. If this model contains a single atom, the program will generate a random poisition for it. If the model contains multiple atoms, the input model will be used as is. After calculation, the resulting model is written to a.res. The starting model, intermediate calculation steps, and the resulting model are also recorded in the history.txt file. The main user interface is s_rap119.py. In cmd window change directory to the folder where a.res etc. are located then type "python s_rap119.py" to solve the structure by the single-atom R1 method. 

Want to try your own data? Rename your data files to a.hkl and a.res. The reflections need to be merged. Typically in cmd window type "python expand_p1bar.py" is sufficient for the required merge. In s_rap119.py the molecular formula need to be edited. Elements are from heavy to light. This makes planning the strategy easier, though not necessary for the calculation. For example, Cl2O2C14. Assume Z=4. In this case, total number of atoms in the cell is 18 x 4 = 72. Set "steps=[4,8,30,72]" is good calculation strategy in this case, means starting from single Cl atom, expand to 4 atoms, then to 8, then to 30, finally to 72 atoms. Edit a.res. Delete any SYMM cards. Use LATT -1. So, it is P1 space group. Use FSAC C H Cl O. Note that in Cl the l must be lower case. Set single Cl atom to start: Cl1   3    0.32  0.4306  0.2176  11.00 0.05      Then you are all set. In cmd window type "python s_rap119.py" to run one cycle of sR1. After delete ghost atoms, run more cycles as needed.

The correct.res contains the correct model. To compare the resulting model in a.res with the correct model, in cmd window type "python compare.py".

Note: you may change s_rap119.py to any filename you like. But a.hkl and a.res are the default names for input and output.

Special note about module pp:  module pp is available from parallelpython.com
Due to some unknown bug in the pp module, the whole program ends with some error messages like 'ERROR: The process "10852" not found.' These can be ignored, and the calculation ends correctly.

To do sR1 calculation you need set fast=2 in s_rap119.py. Usage of some other settings of "fast" parameter is explained below.

About dual-space recycling: You run s_rap119.py with fast=1 to do 2Fo-Fc recycling calculation. No phase refinement implemented. Only some electron density modification via peak picking.

About bond length guided sR1 calculation: run s_rap119.py with fast=200 to perform bond length guided calculation. Suppose you want to add one C atom to atom #10 with bond length between 1.09 A and 1.69 A, add one S atom to atom #20 with bond length between 1.4 A and 2.0 A, you can set cases=[(10,"C",1.39,0.3),(20,"S",1.7,0.3)]. Note that atoms are numbered starting from 1, not from 0, sorry!


About partial-structure R1 (pR1) calculations:

You need some editing work in s_rap119.py, sometime even in tools.py, compare.py.

First, you need to set up a model for the fragment. In tools.py the make_benzene() function is one example of setting up an idealized benzene model:


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


To use this model, in s_rap119.py just uncomment this line:

#fragment0,n_fold=make_benzene()

and change orientation filename in this line:

orientation_file='orientations_benzene.txt'

Now run s_rap119.py with fast=3 once, and run with fast=4 once. To this point, the orientations of benzene rings are saved in orientation_benzene.txt file. 

To add two benzene rings with orientations 0 and 1 to the current partial model, edit these lines in s_rap119.py as:

free_standing=0  # 1: very first 0: add to a partial structure

orientation_selected=[0,1] # orientations are labeled as 0, 1, 2...

If the first ring is really the very first fragment of the model, then use free_standing=1.

You may convert the structure shown in a.res to a model of a fragment. To do this you need to use compare.py. In compare.py there are a set of tools, each is turned on and off by if 1: and if 0: The following section is for making a model:

if 1: # molecular model

    atom_list=read_atoms('a.res')
	
    #save_history(atom_list,runs='starting model')
	
    atoms,labels,s=atomj_solution(atom_list)
	
    for i in range(len(s)):
	
        s[i]=numpy.array(s[i])
		
    p1,p2,p3=s[0],s[3],s[1]
	
    #p3=numpy.array([0.3,0.3,0.3])
	
    A = matrix_A('a.res')
	
    xp,yp,zp=local_xpypzp(p1,p2,p3,A)
	
    for i in range(len(s)):
	
        s[i]=cell_to_local_cartesian(s[i],p1,xp,yp,zp,A)
		
    atom_list=[]
	
    for i in range(len(s)):
	
        x,y,z=s[i]
		
        atom_list.append((atoms[i],labels[i],x,y,z))
		
    with open('C5.txt','w') as f:
	
        for a,l,x,y,z in atom_list:
		
            print(a,l,x,y,z,file=f)

The model will be saved to C5.txt. The line that requires editing is:

    p1,p2,p3=s[0],s[3],s[1]

The local origin will set at p1, x direction is from p1 to p2, y direction is set by vector p1->p3, which is divided into a component along p1->p2 and another component perpendicular to p1->p2, the perpendicular component is the y direction.

So, the above line reads as: p1 uses the first atom s[0], p2 uses the 4th atom s[3], and p3 uses the 2nd atom s[1], in a.res.

To use the model saved in C5.txt you need to edit these lines in s_rap119.py:

orientation_file='orientations_C5.txt'

fragment0,n_fold=make_molecule('C5.txt')

Note that, use the following if you want the inverted version of a model:

fragment0,n_fold=make_invert_molecule('SiPh2tBuMoO4.txt')


You may run s_rap119.py with fast=56 to attach two fragments to the current model such that the local origins of the fragments are attached to the 2nd and 4th atoms. In this case, you need to edit the following section in tools.py:


    if fast==56: # completing model: origin known, 
	
                 # searching orientation
				 
        for i in [2,4]:
		
            atom_list=read_atoms('a.res')
			
            #save_history(atom_list,'starting model',True)

            atoms,labels,s=atomj_solution(atom_list)
			
            s=[numpy.array(p) for p in s]
			
            p=s[i-1]
			
            atom_list=find_fragment_orientations(h,k,l,Fo,A,molecule,Z,atom_list,
			
                fragment0,n_fold,p,orientation_file,
				
                max_orientations=max_orientations,s_angle=s_angle,
				
                starttime=starttime,runs='find fragment orientations')
				
            #atom_list=relax_model(atom_list,f2a,h,k,l,Fo,Fosum,A,content,starttime)
			
            #save_history(arrange(atom_list),'final model',True)
			
            r11=get_sR1(atom_list,h,k,l,f2a,sl,Fo,Fosum,content)
			
            save_history(atom_list,'sR1 = '+str(r11),True)
			
            print('sR1 = '+str(r11))

In this case, only the line 

        for i in [2,4]:

needs editing.

More parameters for pR1 calculations: if you are expecting many possible orientations you may increase max_orientations=10 to max_orientations=1000. You may decrease searching step size from s_angle=20.0 to s_angle=5.0, however, doing that will greatly increase calculation time.

About tweaking the model to optimally agree with experimental data: run s_rap119.py with fast=6. Note that tweaking is automatically performed during the sR1 calculation.

About generating residual reflection intensities: run s_rap119.py with fast=8. Residual hkl data will be saved to a_dif.hkl. Save the original a.hkl as a_original.hkl, save a_dif.hkl as a.hkl to use the residual hkl for searching orientations of a light-atom-only fragment. Remember to save a_original.hkl back as a.hkl before doing further calculations.

Contact Xiaodong Zhang at xzhang2@tulane.edu for further assistance.

