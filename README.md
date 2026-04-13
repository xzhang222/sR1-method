# sR1-method
Python codes for single-atom R1 method

The included a.hkl and a.res provides one example of raw data files. At the start, a.res provide initial model. If this model contains a single atom, the program will generate a random poisition for it. If the model contains multiple atoms, the input model will be used as is. After calculation, the resulting model is written to a.res. The starting model, intermediate calculation steps, and the resulting model are also recorded in the histor.txt file. The main user interface is s_rap119.py. In cmd window type "python s_rap119.py" to solve the structure by the single-atom R1 method. 
