import numpy as np
import random

N=100
maxrange=10000

print('Within the range of [0,10000], we select 100 points.')
print('F1: sin(t)')
print('F2: cos(t)')
#f1=random.sample(range(1, 10000),N)
#f2=random.sample(range(1, 10000),N)
t=np.arange(0,maxrange,N)
f1=np.sin(t)
f2=np.cos(t)


##############################################
f3=[]

for i in range(N):
	com = f1[i] + 1j*f2[i]
	f3.append(com)
##################################3
F3 = np.fft.fft(f3)

F1=[]
F2=[]
for m in range(N):
	f3m=F3[m]
	f3Nm=np.conjugate(F3[N-m-1])

	res1=(f3m+f3Nm)/2
	res2=(f3m-f3Nm)/2j

	F1.append(res1)
	F2.append(res2)

##########################################
print("Fourier transform of f1 :")
print(F1)
print("Fourier transform of f2 :")
print(F2)
##################################