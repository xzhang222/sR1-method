import sys
from tools import *

with open('a_inF.res','r') as f:
	text=f.read()

lines=text.split('\n')
i_FVAR,i_HKLF,i_CELL=0,0,0
for i in range(len(lines)):
	words=lines[i].split()
	if words:
		if words[0].upper()=='FVAR': i_FVAR=i  
		if words[0].upper()=='HKLF': i_HKLF=i 
		if words[0].upper()=='CELL': i_CELL=i 
if not i_FVAR:
	print('missing FVAR')
	sys.exit()
if not i_HKLF:
	print('missing HKLF')
	sys.exit()
if not i_CELL:
	print('missing CELL')
	sys.exit()

A = matrix_A('a_inF.res')
a,b,c=abc(A)
ap=(0,0.5,0.5)
bp=(0.5,0,0.5)
cp=(0.5,0.5,0)
anew=d_exact(ap,A)
bnew=d_exact(bp,A)
cnew=d_exact(cp,A) 
alfanew=degrees(acos(dot(bp,cp,A)/bnew/cnew))
betanew=degrees(acos(dot(ap,cp,A)/anew/cnew))
gammanew=degrees(acos(dot(ap,bp,A)/anew/bnew))
anew,bnew,cnew=round(anew,4),round(bnew,4),round(cnew,4)
alfanew,betanew,gammanew=round(alfanew,3),round(betanew,3),round(gammanew,3)
words=lines[i_CELL].split()
words[2],words[3],words[4]=str(anew),str(bnew),str(cnew)
words[5],words[6],words[7]=str(alfanew),str(betanew),str(gammanew)
print(lines[i_CELL])
lines[i_CELL]=' '.join(words)
print(lines[i_CELL])

converted_lines=lines[:i_FVAR+1]

for line in lines[i_FVAR+1:i_HKLF]:
	words=line.split()
	try:
		x,y,z=float(words[2]),float(words[3]),float(words[4])
		xp,yp,zp=y+z-x,x+z-y,x+y-z
		xp,yp,zp=round(xp,4),round(yp,4),round(zp,4)
		words[2],words[3],words[4]=str(xp),str(yp),str(zp)
		line=' '.join(words)
		converted_lines.append(line)
	except:
		converted_lines.append(line)
converted_lines+=lines[i_HKLF:]

with open('a.res','w') as f:
	for line in converted_lines:
		print(line,file=f)
print('all done')
