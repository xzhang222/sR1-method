# convert from C to P

from read_data import read_hkl3

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

# read in
h,k,l,F2,sigF2 = read_hkl3('a_inC.hkl')

reflections = []
for i in range(len(h)):
	reflections.append((h[i],k[i],l[i],F2[i],sigF2[i]))

reflections.sort(key=lambda r:r[2])
reflections.sort(key=lambda r:r[1])
reflections.sort(key=lambda r:r[0])



print('writing new reflection file a.hkl: ')
with open('a.hkl', 'w') as f:
	for (h,k,l,F2,sigF2) in reflections:
		h,k=(h-k)//2,(h+k)//2
		f.write(st1(h)+st1(k)+st1(l)+st2(F2)+st2(sigF2)+'\n')
	f.write(st1(0)+st1(0)+st1(0)+st2(0)+st2(0)+'\n')

print('done')
