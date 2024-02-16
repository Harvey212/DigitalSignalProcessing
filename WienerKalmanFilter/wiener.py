from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def wiener_filter(img,impulse,err):	
	IMG=np.fft.fft2(img)
	IMP=np.fft.fft2(impulse)#

	see=np.abs(IMP)**2

	app=np.conj(IMP)/(see+err)
	fin=IMG*app
	
	s1=np.fft.ifft2(fin)
	s2=np.fft.fftshift(s1)
	
	return np.abs(s2)

################################################################
img= Image.open('lena.jpg').convert('L')
dd=np.asarray(img.getdata()).reshape(img.size)
################################################################
xx=dd.shape[0]
yy=dd.shape[1]

coo=np.zeros((yy,xx))
coo[int(yy/2):int(yy/2+1),int(xx/2-30/2):int(xx/2+30/2)]=1

p=coo/coo.sum()	
######################3
er=1e-2
#create blurred image
d1=np.fft.fft2(dd)
d2=np.fft.fft2(p)+er
d3=np.fft.ifft2(d1*d2)
blurred=np.abs(np.fft.fftshift(d3))
#################################################################3
#wiener filter
fin=wiener_filter(blurred,p,er)
########################################################
#plt.imshow(fin, cmap = 'gray')
#plt.show()
######################################################
display = [img, blurred, fin]
label = ['Original Image','Blurred', 'After Wiener Filter']

fig = plt.figure(figsize=(12, 10))

for i in range(len(display)):
	fig.add_subplot(1, 3, i+1)
	plt.imshow(display[i], cmap = 'gray')
	plt.title(label[i])

plt.show()


