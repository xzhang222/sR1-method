# expand to Triclinic P1

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
h,k,l,F2,sigF2 = read_hkl3()

# expand Friedel opposites
reflections = []
for i in range(len(h)):
	reflections.append((h[i],k[i],l[i],F2[i],sigF2[i]))
	reflections.append((-h[i],-k[i],-l[i],F2[i],sigF2[i]))

reflections.sort(key=lambda r:r[2])
reflections.sort(key=lambda r:r[1])
reflections.sort(key=lambda r:r[0])

# merge
n = len(reflections)
print(n)
reflects = []
i = 0
while i < n:
	h,k,l,F2,sigF2 = reflections[i]
	i+=1
	count = 0
	for j in range(i, n):
		hh,kk,ll,F22,sigF22 = reflections[i]
		if h==hh and k==kk and l==ll:
			count += 1
			i += 1
			F2 += F22  
			sigF2 += sigF22
		else:
			break
	F2 /= count+1
	sigF2 /= count+1
	reflects.append((h,k,l,round(F2,2),round(sigF2,2)))
print(len(reflects))


print('writing new reflection file a.hkl: ')
with open('a.hkl', 'w') as f:
	for (h,k,l,F2,sigF2) in reflects:
		f.write(st1(h)+st1(k)+st1(l)+st2(F2)+st2(sigF2)+'\n')
	f.write(st1(0)+st1(0)+st1(0)+st2(0)+st2(0)+'\n')

print('done')
