import numpy as np
import random
import math
##############################
N=21
k=int((N-1)/2)
k_2=k+2
fs=8000
################
#transition band:1600~2000
#transst=1600/fs
#transed=2000/fs
####################################
#stopband: 0~1800   weight:0.8
stopst=0
stoped=1600
stopweight=0.8
###################################
#passband:1800~4000    weight:1
passst=2000
passed=4000
passweight=1
##################
delta=0.0001
##################
F=[]
ini=int(k_2/2)
rest=k_2-ini

s1=random.sample(range(stopst,stoped),ini)
s1=sorted(s1)
for i in range(len(s1)):
	F.append(s1[i]/fs)
###############################
s2=random.sample(range(passst+1,passed+1),rest)
s2=sorted(s2)
for i in range(len(s2)):
	F.append(s2[i]/fs)
###################################
accE=[]
check=True
preE=math.inf
########################################
while check:
	M=np.zeros((k_2,k_2))
	H=[]

	for i in range(k_2):
		for j in range(k+1):
			M[i,j]=math.cos(2*math.pi*j*F[i])

	for i in range(k_2):
		if F[i]*fs<stoped+1:
			weight=stopweight
			H.append(0)
		else:
			weight=passweight
			H.append(1)		

		M[i,k+1]=(1/weight)*pow(-1,i)
	####################################
	S=np.matmul(np.linalg.inv(M),np.array(H).transpose())
	#############################
	Errs=[]
	####################################################
	for i in range(passed+1):
		temp=0
		if ((i>stopst-1) and (i<stoped)):
			for j in range(k+1):
				temp+=S[j]*math.cos(2*math.pi*j*(i/fs))
			H=0
			W=stopweight
		elif ((i>passst) and (i<passed+1)):
			for j in range(k+1):
				temp+=S[j]*math.cos(2*math.pi*j*(i/fs))
			H=1
			W=passweight
		else:
			W=0

		err=(temp-H)*W
		Errs.append(err)

	###############################################
	extremeF=[]
	extremeE=[]

	boundaryF=[]
	boundaryE=[]

	for i in range(passed+1):
		if ((i>stopst) and (i<stoped)) or ((i>passst) and (i<passed)):
			if ((Errs[i]>Errs[i-1]) and (Errs[i]>Errs[i+1])) or ((Errs[i]<Errs[i-1]) and (Errs[i]<Errs[i+1])):
				extremeF.append(i/fs)
				extremeE.append(abs(Errs[i]))
		elif (i==stopst) or (i==passst):
			if (Errs[i]>0) and (Errs[i]>Errs[i+1]):
				boundaryF.append(i/fs)
				boundaryE.append(abs(Errs[i]))
			elif (Errs[i]<0) and (Errs[i]<Errs[i+1]):
				boundaryF.append(i/fs)
				boundaryE.append(abs(Errs[i]))
		elif (i==stoped) or (i==passed):
			if (Errs[i]>0) and (Errs[i-1]<Errs[i]):
				boundaryF.append(i/fs)
				boundaryE.append(abs(Errs[i]))
			elif (Errs[i]<0) and (Errs[i-1]>Errs[i]):
				boundaryF.append(i/fs)
				boundaryE.append(abs(Errs[i]))

	candyF=extremeF
	candyE=extremeE
	###########################
	#print('##################')
	#print(len(extremeE))
	
	##############################
	if len(extremeE)<k_2:
		lastn=k_2-len(extremeE)
		candyind=np.argsort(boundaryE)[::-1][:lastn]

		for i in range(len(candyind)):
			myind=candyind[i]
			myF=boundaryF[myind]
			myE=boundaryE[myind]
			candyF.append(myF)
			candyE.append(myE)
	#########################################
	#print(len(candyF))
	#print('####################')

	finind=np.argsort(candyE)[::-1]
	bg=finind[0]
	nowE=candyE[bg]
	accE.append(nowE)
	####################################
	idd=np.argsort(candyF)
	finF=[]
	for i in range(len(idd)):
		ii=idd[i]
		finF.append(candyF[ii])

	if ((preE-nowE)>delta) or ((preE-nowE)<0):
		F=finF
		preE=nowE
	else:
		check=False
		finS=S
		R=[]

		for p in range(len(finF)):
			ff=finF[p]
			temp=0
			for w in range(k+1):
				temp+=finS[w]*math.cos(2*math.pi*w*ff)
			R.append(temp)
############################################
h=np.zeros(N)
h[k]=finS[0]
for n in range(1,k+1):
	h[k+n]=finS[n]/2
	h[k-n]=finS[n]/2
#########################################
print('the frequency response')
print(R)
###################################
print('the impulse response h[n]')
print(h)
###################################
print('the maximal error for each iteration')
print(accE)