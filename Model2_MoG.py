

 import cv2\import numpy as np\
from scipy.stats import multivariate_normal\
import gc\
from decimal import Decimal\
import matplotlib.pyplot as plt\
\
gc.collect()\
\
\
I=1000\
D=100\
K=2\
imageVector = np.empty((1000,100))\
imageVector1 = np.empty((1000,100))\
imageVector2 = np.empty((1000,100))\
imageVector3 = np.empty((100,100))\
imageVector4 = np.empty((100,100))\
test = np.empty(100)\
data_meanVector = np.empty(100)\
cov1 = np.empty((100,100))\
cov2 = np.empty(100)\
sig= np.empty((100,100,100))\
\
var = np.empty(100)\
meanImage_face = np.empty((10,10))\
covImage_face = np.empty((10,10))\
meanImage_non_face = np.empty((10,10))\
covImage_non_face = np.empty((10,10))\
\
for i in range(I):\
	img = cv2.imread("/Users/binoythomas/Desktop/untitled folder/"+str(i+1)+str(1)+".jpg")\
	im_resized = cv2.resize(img, (10, 10), interpolation=cv2.INTER_LINEAR)\
	image = cv2.cvtColor(im_resized, cv2.COLOR_RGB2GRAY)\
	imageVector1[i][:] = image.flatten()\
\
\
for i in range(I):\
	img = cv2.imread("/Users/binoythomas/Desktop/untitled folder 5/"+str(i+1)+str(1)+".jpg")\
	im_resized = cv2.resize(img, (10, 10), interpolation=cv2.INTER_LINEAR)\
	image = cv2.cvtColor(im_resized, cv2.COLOR_RGB2GRAY)\
	imageVector2[i][:] = image.flatten()\
    \
for i in range(100):\
	img = cv2.imread("/Users/binoythomas/Desktop/untitled folder 3/"+str(i+1)+str(1)+".jpg")\
	im_resized = cv2.resize(img, (10, 10), interpolation=cv2.INTER_LINEAR)\
	image = cv2.cvtColor(im_resized, cv2.COLOR_RGB2GRAY)\
	imageVector3[i][:] = image.flatten()\
\
for i in range(100):\
	img = cv2.imread("/Users/binoythomas/Desktop/untitled folder 4/"+str(i+1)+str(1)+".jpg")\
	im_resized = cv2.resize(img, (10, 10), interpolation=cv2.INTER_LINEAR)\
	image = cv2.cvtColor(im_resized, cv2.COLOR_RGB2GRAY)\
	imageVector4[i][:] = image.flatten()\
\
temp_variance= np.empty((1,100))\
data_variance= np.zeros((100, 100))\
previous_L = 1000000\
precision = 0.01\
meanVector= np.random.randint(256, size=(K, D))\
data_meanVector = np.mean(imageVector1,axis=0)\
meanVector= np.random.randint(256, size=(K, D))\
data_meanVector = np.mean(imageVector1,axis=0)\
\
lambda1=np.full((K), 0.5)\
\
for i in range(I):\
	temp_variance[:] = imageVector1[i][:] - data_meanVector[:]\
	temp_variance = np.matmul(temp_variance.T,temp_variance)\
	data_variance = np.add(data_variance,temp_variance)\
\
data_variance= data_variance/I\
\
for k in range(K):\
	sig[:][:][k] = data_variance[:][:]\
\
count =0\
while count!=40:\
	count+=1\
	l=np.zeros((K,I))\
	r=np.zeros((K,I))\
	s=np.zeros((I))\
\
	for k in range(K):\
		l[k,:]=lambda1[k] * multivariate_normal.pdf(imageVector1, meanVector[k,:], sig[:][:][k])\
		\
\
	s= l.sum(axis=0)\
	for i in range(I):\
		r[:,i] = l[:,i] / s[i]\
	\
	r_summed_rows = r.sum(axis=1)\
	r_summed_all = r_summed_rows.sum(axis=0)\
	for k in range (K):\
\
		lambda1[k] = r_summed_rows[k] / r_summed_all\
		new_meanVector = np.zeros((D))\
\
		for i in range(I):\
			new_meanVector = new_meanVector + r[k,i]*imageVector1[i,:]\
\
\
		\
		meanVector[k,:] = new_meanVector / r_summed_rows[k]\
		new_sigma = np.zeros((D,D))\
		for i in range(I):\
			temp_variance[:] = imageVector1[i][:] - meanVector[k][:]\
			temp_variance = r[k,i]*np.matmul(temp_variance.T,temp_variance)\
			new_sigma = np.add(new_sigma,temp_variance)\
\
\
		sig[:][:][k] = new_sigma[:][:] / r_summed_rows[k]\
	print(sig)\
	temp = np.empty ((K,I))\
	for k in range(K):\
		temp[k,:]=lambda1[k]*multivariate_normal.pdf(imageVector1, meanVector[k,:], sig[:][:][k])\
\
 \
	temp = temp.sum(axis=0)\
	temp1 = np.log(temp)\
	L = temp1.sum()\
	if abs(L - previous_L) < precision:\
		print (count)\
		break \
\
	previous_L = L\
\
\
\
temp_variance= np.empty((1,100))\
data_variance= np.zeros((100, 100))\
previous_L = 1000000\
precision = 0.01\
\
meanVector= np.random.randint(256, size=(K, D))\
data_meanVector = np.mean(imageVector2,axis=0)\
#print(data_meanVector)\
lambda2=np.full((K), 0.5)\
\
for i in range(I):\
	temp_variance[:] = imageVector2[i][:] - data_meanVector[:]\
	temp_variance = np.matmul(temp_variance.T,temp_variance)\
	data_variance = np.add(data_variance,temp_variance)\
\
data_variance= data_variance/I\
#print(data_variance)\
for k in range(K):\
	sig[:][:][k] = data_variance[:][:]\
print(sig)\
\
count =0\
while count!=40:\
	count+=1\
	l=np.zeros((K,I))\
	r=np.zeros((K,I))\
	s=np.zeros((I))\
	for k in range(K):\
		l[k,:]=lambda2[k] * multivariate_normal.pdf(imageVector2, meanVector[k,:], sig[:][:][k])\
		\
	#print(l)\
	s= l.sum(axis=0)\
	for i in range(I):\
		r[:,i] = l[:,i] / s[i]\
	\
	r_summed_rows = r.sum(axis=1)\
	r_summed_all = r_summed_rows.sum(axis=0)\
	for k in range (K):\
\
		lambda2[k] = r_summed_rows[k] / r_summed_all\
		new_meanVector = np.zeros((D))\
\
		for i in range(I):\
			new_meanVector = new_meanVector + r[k,i]*imageVector2[i,:]\
\
\
		\
		meanVector[k,:] = new_meanVector / r_summed_rows[k]\
		new_sigma = np.zeros((D,D))\
		for i in range(I):\
			temp_variance[:] = imageVector2[i][:] - meanVector[k][:]\
			temp_variance = r[k,i]*np.matmul(temp_variance.T,temp_variance)\
			new_sigma = np.add(new_sigma,temp_variance)\
\
\
		sig[:][:][k] = new_sigma[:][:] / r_summed_rows[k]\
	print(r_summed_rows)\
	#print(sig)\
	temp = np.empty ((K,I))\
	for i in range(100):\
		for j in range(100):\
			for w in range(100):\
				if np.isnan(sig[i][j][w])=='True':\
					sig[i][j][w]=0\
	for k in range(K):\
		temp[k,:]=lambda2[k]*multivariate_normal.pdf(imageVector2, meanVector[k,:], sig[:][:][k])\
\
 \
	temp = temp.sum(axis=0)\
	temp1 = np.log(temp)\
	L = temp1.sum()\
	if abs(L - previous_L) < precision:\
		print (count)\
		break \
\
	previous_L = L\
\
\
\
temp_variance= np.empty((1,100))\
data_variance= np.zeros((100, 100))\
previous_L = 1000000\
precision = 0.01\
\
meanVector= np.random.randint(256, size=(K, D))\
data_meanVector = np.mean(imageVector2,axis=0)\
#print(data_meanVector)\
lambda2=np.full((K), 0.5)\
\
for i in range(I):\
	temp_variance[:] = imageVector2[i][:] - data_meanVector[:]\
	temp_variance = np.matmul(temp_variance.T,temp_variance)\
	data_variance = np.add(data_variance,temp_variance)\
\
data_variance= data_variance/I\
#print(data_variance)\
for k in range(K):\
	sig[:][:][k] = data_variance[:][:]\
print(sig)\
\
count =0\
while count!=40:\
	count+=1\
	l=np.zeros((K,I))\
	r=np.zeros((K,I))\
	s=np.zeros((I))\
	for k in range(K):\
		l[k,:]=lambda2[k] * multivariate_normal.pdf(imageVector2, meanVector[k,:], sig[:][:][k])\
		\
	#print(l)\
	s= l.sum(axis=0)\
	for i in range(I):\
		r[:,i] = l[:,i] / s[i]\
	\
	r_summed_rows = r.sum(axis=1)\
	r_summed_all = r_summed_rows.sum(axis=0)\
	for k in range (K):\
\
		lambda2[k] = r_summed_rows[k] / r_summed_all\
		new_meanVector = np.zeros((D))\
\
		for i in range(I):\
			new_meanVector = new_meanVector + r[k,i]*imageVector2[i,:]\
\
\
		\
		meanVector[k,:] = new_meanVector / r_summed_rows[k]\
		new_sigma = np.zeros((D,D))\
		for i in range(I):\
			temp_variance[:] = imageVector2[i][:] - meanVector[k][:]\
			temp_variance = r[k,i]*np.matmul(temp_variance.T,temp_variance)\
			new_sigma = np.add(new_sigma,temp_variance)\
\
\
		sig[:][:][k] = new_sigma[:][:] / r_summed_rows[k]\
	print(r_summed_rows)\
	#print(sig)\
	temp = np.empty ((K,I))\
	for i in range(100):\
		for j in range(100):\
			for w in range(100):\
				if np.isnan(sig[i][j][w])=='True':\
					sig[i][j][w]=0\
	for k in range(K):\
		temp[k,:]=lambda2[k]*multivariate_normal.pdf(imageVector2, meanVector[k,:], sig[:][:][k])\
\
 \
	temp = temp.sum(axis=0)\
	temp1 = np.log(temp)\
	L = temp1.sum()\
	if abs(L - previous_L) < precision:\
		print (count)\
		break \
\
	previous_L = L\
\
temp_variance= np.empty((1,100))\
data_variance= np.zeros((100, 100))\
previous_L = 1000000\
precision = 0.01\
\
meanVector= np.random.randint(256, size=(K, D))\
data_meanVector = np.mean(imageVector2,axis=0)\
#print(data_meanVector)\
lambda2=np.full((K), 0.5)\
\
for i in range(I):\
	temp_variance[:] = imageVector2[i][:] - data_meanVector[:]\
	temp_variance = np.matmul(temp_variance.T,temp_variance)\
	data_variance = np.add(data_variance,temp_variance)\
\
data_variance= data_variance/I\
#print(data_variance)\
for k in range(K):\
	sig[:][:][k] = data_variance[:][:]\
print(sig)\
\
count =0\
while count!=40:\
	count+=1\
	l=np.zeros((K,I))\
	r=np.zeros((K,I))\
	s=np.zeros((I))\
	for k in range(K):\
		l[k,:]=lambda2[k] * multivariate_normal.pdf(imageVector2, meanVector[k,:], sig[:][:][k])\
		\
	#print(l)\
	s= l.sum(axis=0)\
	for i in range(I):\
		r[:,i] = l[:,i] / s[i]\
	\
	r_summed_rows = r.sum(axis=1)\
	r_summed_all = r_summed_rows.sum(axis=0)\
	for k in range (K):\
\
		lambda2[k] = r_summed_rows[k] / r_summed_all\
		new_meanVector = np.zeros((D))\
\
		for i in range(I):\
			new_meanVector = new_meanVector + r[k,i]*imageVector2[i,:]\
\
\
		\
		meanVector[k,:] = new_meanVector / r_summed_rows[k]\
		new_sigma = np.zeros((D,D))\
		for i in range(I):\
			temp_variance[:] = imageVector2[i][:] - meanVector[k][:]\
			temp_variance = r[k,i]*np.matmul(temp_variance.T,temp_variance)\
			new_sigma = np.add(new_sigma,temp_variance)\
\
\
		sig[:][:][k] = new_sigma[:][:] / r_summed_rows[k]\
	print(r_summed_rows)\
	#print(sig)\
	temp = np.empty ((K,I))\
	for i in range(100):\
		for j in range(100):\
			for w in range(100):\
				if np.isnan(sig[i][j][w])=='True':\
					sig[i][j][w]=0\
	for k in range(K):\
		temp[k,:]=lambda2[k]*multivariate_normal.pdf(imageVector2, meanVector[k,:], sig[:][:][k])\
\
 \
	temp = temp.sum(axis=0)\
	temp1 = np.log(temp)\
	L = temp1.sum()\
	if abs(L - previous_L) < precision:\
		print (count)\
		break \
\
	previous_L = L\
\
for count in range(101):\
	for n in range(100):\
		prob_1 = lambda1[0]*multivariate_normal.pdf(imageVector3[n][:], mu1[0][:], sig1[:][:][0]) + lambda1[1]*multivariate_normal.pdf(imageVector3[n][:], mu1[1][:],sig1[:][:][1])\
		prob_2 = lambda1[0]*multivariate_normal.pdf(imageVector3[n][:], mu2[0][:], sig1[:][:][0]) + lambda1[1]*multivariate_normal.pdf(imageVector3[n][:], mu2[1][:],sig1[:][:][1])\
		probVectorFace[n] = prob_1/(prob_1+prob_2)\
        \
		if probVectorFace[n] > (count*0.01):\
			TP[count] +=1\
		else:\
			FN[count] +=1 \
\
for count in range(101):\
	for n in range(100):\
		prob_1 = lambda1[0]*multivariate_normal.pdf(imageVector4[n][:], mu1[0][:], sig1[:][:][0]) + lambda1[1]*multivariate_normal.pdf(imageVector4[n][:], mu1[1][:],sig1[:][:][1])\
		prob_2 = lambda1[0]*multivariate_normal.pdf(imageVector4[n][:], mu2[0][:], sig1[:][:][0]) + lambda1[1]*multivariate_normal.pdf(imageVector4[n][:], mu2[1][:],sig1[:][:][1])\
		probVectorNonFace[n] = prob_2/(prob_1+prob_2)\
		if probVectorNonFace[n] > (count*0.01):\
			TN[count] +=1 \
		else:\
			FP[count] +=1\
\
FPR_0_5 = FP[50] / 100\
FNR_0_5 = FN[50] / 100\
MCR_0_5 = (FP[50]+FN[50]) / 200\
\
print (FPR_0_5)\
print (FNR_0_5)\
print (MCR_0_5)\
\
\
TPR = TP/100\
FPR = FP/100\
\
\
\
plt.plot(np.flipud(FPR),TPR)\
plt.xlabel('False Positive Rate')\
plt.ylabel('True Positive Rate')\
plt.title("ROC Curve Mixture of Gaussian model")\
plt.xlim([0, 1])\
plt.ylim([0, 1])\
plt.show()\
\
\
\
}
