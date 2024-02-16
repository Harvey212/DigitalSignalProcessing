import numpy as np 
import math
import cmath
import matplotlib.pyplot as plt

def filter(F):
	return 1j*2*math.pi*F

##########################
#F=-0.5: H(-0.5) = -1*j*pi
#F=0: H(0) = 0
#F=0.5: H(0.5) = j*pi 
#########################
k = 100
##############################
N=np.linspace(0,1,2*k+1)
pt=0
hd=[]
for m in range(len(N)):
	

	if m>k:
		#k+1~N-1 =>F= (>-0.5) ~ (0-)
		pt =(m-len(N))/len(N)
	else:
		#0~k => F=0~ (<0.5)
		pt = m/len(N)

	hd.append(filter(pt))
###############################
#0:0
#k:j pi
#k+1:-j pi
#N-1:0-
#######################################
#to reduce transition band error
hd[k] = 0.5j*math.pi
hd[k+1] = -0.5j*math.pi
#####################################
r1n=np.fft.ifft(hd)
rn=np.concatenate((r1n[k:],r1n[0:k]))
hn=rn

plt.plot(hn.real)

plt.xlabel('N')
plt.ylabel('amplitude')
plt.title('impulse response')
plt.show()
#####################################
f=np.linspace(0,1,10001)
RF =[]
for i in range(len(f)):
	F=f[i]
	rf = 0
	for m in range(len(rn)):
		rf+=rn[m]*cmath.exp(-2j*math.pi*F*(m-k-1))

	RF.append(rf)


plt.plot(f,np.array(RF).imag)
plt.plot(N,np.array(hd).imag,'r--')
plt.legend(['R(F)','Hd(F)'])
plt.title('frequency response')
plt.xlabel('F')
plt.ylabel('amplitude')
plt.show()