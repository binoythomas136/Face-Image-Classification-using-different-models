import cv2
import numpy as np
from scipy.stats import multivariate_normal
import gc
import matplotlib.pyplot as plt

gc.collect()


images=1000
size=100
image_matrix = np.empty((1000,100))
training_face = np.empty((1000,100))
training_non_face = np.empty((1000,100))
testing_face = np.empty((100,100))
testing_non_face = np.empty((100,100))
test = np.empty(100)

mean_matrix = np.empty(100)

def gaussian_calculate(image_matrix):


	mean_matrix = image_matrix.mean(axis=0)


	variance_matrix = np.cov(image_matrix,rowvar=False)

	variance_matrix = np.diag(np.diag(variance_matrix))

	return mean_matrix, variance_matrix


for i in range(images):
	img = cv2.imread("/Users/binoythomas/Desktop/untitled folder/"+str(i+1)+str(1)+".jpg")
	im_resized = cv2.resize(img, (10, 10), interpolation=cv2.INTER_LINEAR)
	image = cv2.cvtColor(im_resized, cv2.COLOR_RGB2GRAY)
	training_face[i][:] = image.flatten()


for i in range(images):
	img = cv2.imread("/Users/binoythomas/Desktop/untitled folder 2/"+str(i+1)+str(1)+".jpg")
	im_resized = cv2.resize(img, (10, 10), interpolation=cv2.INTER_LINEAR)
	image = cv2.cvtColor(im_resized, cv2.COLOR_RGB2GRAY)
	training_non_face[i][:] = image.flatten()


mu1, sig1 = gaussian_calculate(training_face)
mu2, sig2 = gaussian_calculate(training_non_face)


for i in range(100):
	img = cv2.imread("/Users/binoythomas/Desktop/untitled folder 3/"+str(i+1)+str(1)+".jpg")
	im_resized = cv2.resize(img, (10, 10), interpolation=cv2.INTER_LINEAR)
	image = cv2.cvtColor(im_resized, cv2.COLOR_RGB2GRAY)
	testing_face[i][:] = image.flatten()

for i in range(100):
	img = cv2.imread("/Users/binoythomas/Desktop/untitled folder 4/"+str(i+1)+str(1)+".jpg")
	im_resized = cv2.resize(img, (10, 10), interpolation=cv2.INTER_LINEAR)
	image = cv2.cvtColor(im_resized, cv2.COLOR_RGB2GRAY)
	testing_non_face[i][:] = image.flatten()
    
true_positive=np.zeros((101,1))
true_negative=np.zeros((101,1))
false_positive=np.zeros((101,1))
false_negative=np.zeros((101,1))

face_prob_matrix = np.empty((100))
non_face_prob_matrix = np.empty((100))

true_positive_rate=np.zeros((101,1))
false_postive_rate=np.zeros((101,1))


import cv2
import numpy as np
from scipy.stats import multivariate_normal
import gc
import matplotlib.pyplot as plt

gc.collect()


images=1000
size=100
image_matrix = np.empty((1000,100))
training_face = np.empty((1000,100))
training_non_face = np.empty((1000,100))
testing_face = np.empty((100,100))
testing_non_face = np.empty((100,100))
test = np.empty(100)

mean_matrix = np.empty(100)

def gaussian_calculate(image_matrix):


	mean_matrix = image_matrix.mean(axis=0)


	variance_matrix = np.cov(image_matrix,rowvar=False)

	variance_matrix = np.diag(np.diag(variance_matrix))

	return mean_matrix, variance_matrix


for i in range(images):
	img = cv2.imread("/Users/binoythomas/Desktop/untitled folder/"+str(i+1)+str(1)+".jpg")
	im_resized = cv2.resize(img, (10, 10), interpolation=cv2.INTER_LINEAR)
	image = cv2.cvtColor(im_resized, cv2.COLOR_RGB2GRAY)
	training_face[i][:] = image.flatten()


for i in range(images):
	img = cv2.imread("/Users/binoythomas/Desktop/untitled folder 2/"+str(i+1)+str(1)+".jpg")
	im_resized = cv2.resize(img, (10, 10), interpolation=cv2.INTER_LINEAR)
	image = cv2.cvtColor(im_resized, cv2.COLOR_RGB2GRAY)
	training_non_face[i][:] = image.flatten()


mu1, sig1 = gaussian_calculate(training_face)
mu2, sig2 = gaussian_calculate(training_non_face)


for i in range(100):
	img = cv2.imread("/Users/binoythomas/Desktop/untitled folder 3/"+str(i+1)+str(1)+".jpg")
	im_resized = cv2.resize(img, (10, 10), interpolation=cv2.INTER_LINEAR)
	image = cv2.cvtColor(im_resized, cv2.COLOR_RGB2GRAY)
	testing_face[i][:] = image.flatten()

for i in range(100):
	img = cv2.imread("/Users/binoythomas/Desktop/untitled folder 4/"+str(i+1)+str(1)+".jpg")
	im_resized = cv2.resize(img, (10, 10), interpolation=cv2.INTER_LINEAR)
	image = cv2.cvtColor(im_resized, cv2.COLOR_RGB2GRAY)
	testing_non_face[i][:] = image.flatten()
    
true_positive=np.zeros((101,1))
true_negative=np.zeros((101,1))
false_positive=np.zeros((101,1))
false_negative=np.zeros((101,1))

face_prob_matrix = np.empty((100))
non_face_prob_matrix = np.empty((100))

true_positive_rate=np.zeros((101,1))
false_postive_rate=np.zeros((101,1))

import cv2
import numpy as np
from scipy.stats import multivariate_normal
import gc
import matplotlib.pyplot as plt

gc.collect()


images=1000
size=100
image_matrix = np.empty((1000,100))
training_face = np.empty((1000,100))
training_non_face = np.empty((1000,100))
testing_face = np.empty((100,100))
testing_non_face = np.empty((100,100))
test = np.empty(100)

mean_matrix = np.empty(100)

def gaussian_calculate(image_matrix):


	mean_matrix = image_matrix.mean(axis=0)


	variance_matrix = np.cov(image_matrix,rowvar=False)

	variance_matrix = np.diag(np.diag(variance_matrix))

	return mean_matrix, variance_matrix


for i in range(images):
	img = cv2.imread("/Users/binoythomas/Desktop/untitled folder/"+str(i+1)+str(1)+".jpg")
	im_resized = cv2.resize(img, (10, 10), interpolation=cv2.INTER_LINEAR)
	image = cv2.cvtColor(im_resized, cv2.COLOR_RGB2GRAY)
	training_face[i][:] = image.flatten()


for i in range(images):
	img = cv2.imread("/Users/binoythomas/Desktop/untitled folder 2/"+str(i+1)+str(1)+".jpg")
	im_resized = cv2.resize(img, (10, 10), interpolation=cv2.INTER_LINEAR)
	image = cv2.cvtColor(im_resized, cv2.COLOR_RGB2GRAY)
	training_non_face[i][:] = image.flatten()


mu1, sig1 = gaussian_calculate(training_face)
mu2, sig2 = gaussian_calculate(training_non_face)


for i in range(100):
	img = cv2.imread("/Users/binoythomas/Desktop/untitled folder 3/"+str(i+1)+str(1)+".jpg")
	im_resized = cv2.resize(img, (10, 10), interpolation=cv2.INTER_LINEAR)
	image = cv2.cvtColor(im_resized, cv2.COLOR_RGB2GRAY)
	testing_face[i][:] = image.flatten()

for i in range(100):
	img = cv2.imread("/Users/binoythomas/Desktop/untitled folder 4/"+str(i+1)+str(1)+".jpg")
	im_resized = cv2.resize(img, (10, 10), interpolation=cv2.INTER_LINEAR)
	image = cv2.cvtColor(im_resized, cv2.COLOR_RGB2GRAY)
	testing_non_face[i][:] = image.flatten()
    
true_positive=np.zeros((101,1))
true_negative=np.zeros((101,1))
false_positive=np.zeros((101,1))
false_negative=np.zeros((101,1))

face_prob_matrix = np.empty((100))
non_face_prob_matrix = np.empty((100))

true_positive_rate=np.zeros((101,1))
false_postive_rate=np.zeros((101,1))

FPR_0_5 = false_positive[50] / 100
FNR_0_5 = false_negative[50] / 100
MCR_0_5 = (false_positive[50]+false_negative[50]) / 200

print (FPR_0_5)
print (FNR_0_5)
print (MCR_0_5)


TPR = true_positive/100
FPR = false_positive/100


plt.plot(np.flipud(FPR),TPR)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve Gaussian model")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()
