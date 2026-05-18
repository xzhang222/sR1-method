# sR1-method
Python codes for single-atom R1 method

The included a.hkl and a.res provide one example of raw data files. At the start, a.res provides initial model. If this model contains a single atom, the program will generate a random poisition for it. If the model contains multiple atoms, the input model will be used as is. After calculation, the resulting model is written to a.res. The starting model, intermediate calculation steps, and the resulting model are also recorded in the history.txt file. The main user interface is s_rap119.py. In cmd window change directory to the folder where a.res etc. are located then type "python s_rap119.py" to solve the structure by the single-atom R1 method. 

Want to try your own data? Rename your data files to a.hkl and a.res. The reflections need to be merged. Typically in cmd window type "python expand_p1bar.py" is sufficient for the required merge. In s_rap119.py the molecular formula need to be edited. Elements are from heavy to light. For example, Cl2O2C14. Assume Z=4. In this case, total number of atoms in the cell is 18 x 4 = 72. Set "steps=[4,8,30,72]" is good calculation strategy in this case, means starting from single Cl atom, expand to 4 atoms, then to 8, then to 30, finally to 72 atoms. Edit a.res. Delete any SYMM cards. Use LATT -1. So, it is P1 space group. Use FSAC C H Cl O. Note C always at #1, H always at #2, rest elements Cl and O are in order from heavy to light. Note that in Cl the l must be lower case. Set single Cl atom to start: Cl1   3    0.32  0.4306  0.2176  11.00 0.05      Then you are all set. In cmd window type "python s_rap119.py" to run one cycle of sR1. After delete ghost atoms, run more cycles as needed.

The correct.res contains the correct model. To compare the resulting model in a.res with the correct model, in cmd window type "python compare.py".

Note: you may change s_rap119.py to any filename you like. But a.hkl and a.res are the default names for input and output.

Special note about module pp:  module pp is available from parallelpython.com

