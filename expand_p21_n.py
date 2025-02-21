# expand Orthorhombic Pnma to Triclinic P1

from read_data import read_hkl3
import sys

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

h,k,l,F2,sigF2 = read_hkl3()

reflections = []
for i in range(len(h)):
	reflections.append((h[i],k[i],l[i],F2[i],sigF2[i]))

reflections.sort(key=lambda r:r[2])
reflections.sort(key=lambda r:r[1])
reflections.sort(key=lambda r:r[0])

# for P2(1)/n: hkl eq h-kl
print('writing new reflection file a.hkl: ')
with open('a.hkl', 'w') as f:
	for (h,k,l,F2,sigF2) in reflections:
		f.write(st1(h)+st1(k)+st1(l)+st2(F2)+st2(sigF2)+'\n')
		f.write(st1(h)+st1(-k)+st1(l)+st2(F2)+st2(sigF2)+'\n')
		f.write(st1(-h)+st1(-k)+st1(-l)+st2(F2)+st2(sigF2)+'\n')
		f.write(st1(-h)+st1(k)+st1(-l)+st2(F2)+st2(sigF2)+'\n')

print('done')
