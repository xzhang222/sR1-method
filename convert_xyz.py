import sys

fname=input('type in file name: ')
with open(fname,'r') as f:
	text=f.read()

lines=text.split('\n')
i_FVAR,i_HKLF=0,0
for i in range(len(lines)):
	words=lines[i].split()
	if words:
		if words[0].upper()=='FVAR': i_FVAR=i  
		if words[0].upper()=='HKLF': i_HKLF=i 
if not i_FVAR:
	print('missing FVAR')
	sys.exit()
if not i_HKLF:
	print('missing HKLF')
	sys.exit()

converted_lines=lines[:i_FVAR+1]

for line in lines[i_FVAR+1:i_HKLF]:
	words=line.split()
	try:
		x,y,z=float(words[2]),float(words[3]),float(words[4])
		xp,yp,zp=x-2,y+1,z 
		words[2],words[3],words[4]=str(xp),str(yp),str(zp)
		line=' '.join(words)
		converted_lines.append(line)
	except:
		converted_lines.append(line)
converted_lines+=lines[i_HKLF:]
#fname='new.res'
with open(fname,'w') as f:
	for line in converted_lines:
		print(line,file=f)
print('all done')
