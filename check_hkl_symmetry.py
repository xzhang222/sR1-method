from cmath import exp, pi, phase
from math import degrees

#orthorhombic Pnma
# symmetries = [
# 'x,y,z',
# '1/2-x,-y,1/2+z',
# '-x,1/2+y,-z',
# '1/2+x,1/2-y,1/2-z',
# '-x,-y,-z',
# '1/2+x,y,1/2-z',
# 'x,1/2-y,z',
# '1/2-x,1/2+y,1/2+z',
# ]

#orthorhombic C2/c
# symmetries = [
# 'x,y,z',
# 'x+1/2,y+1/2,z',
# '-x,y,1/2-z',
# '1/2-x,y+1/2,1/2-z',
# '-x,-y,-z',
# '1/2-x,1/2-y,-z',
# 'x,-y,1/2+z',
# 'x+1/2,1/2-y,z+1/2'
# ]

#orthorhombic Pmn2(1)
# symmetries = [
# 'x,y,z',
# '1/2-x,-y,1/2+z',
# '1/2+x,-y,1/2+z',
# '-x,y,z'
# ]

# P2(1)
# symmetries = [
# 'x,y,z',
# '-x,1/2+y,-z'
# ]

# Pn
# symmetries = [
# 'x,y,z',
# '1/2+x,-y,1/2+z'
# ]

# P2(1)/m
# symmetries = [
# 'x,y,z',
# '-x,1/2+y,-z',
# '-x,-y,-z',
# 'x,1/2-y,z'
# ]

# P2(1)/c  
# symmetries = [
# 'x,y,z',
# '-x,1/2+y,1/2-z',
# '-x,-y,-z',
# 'x,1/2-y,1/2+z'
# ]

#P2(1)/n
symmetries = [
'x,y,z',
'0.5-x,0.5+y,0.5-z',
'-x,-y,-z',
'0.5+x,0.5-y,0.5+z'
]

# I-4
# symmetries = [
# 'x,y,z',
# '-x,-y,z',
# 'y,-x,-z',
# '-y,x,-z'
# ]


h0,k0,l0 = 2,3,7
x,y,z = 0.1321,0.2436,0.3763

s = [1,-1]
for sh in s:
	for sk in s:
		for sl in s:
			h,k,l = sh*h0,sk*k0,sl*l0
			Fhkl = 0
			for sym in symmetries:
				x0,y0,z0 = eval(sym)
				Fhkl += exp(1j*2*pi*(h*x0+k*y0+l*z0))

			print(h,k,l,round(abs(Fhkl),2),round(degrees(phase(Fhkl)),2))
