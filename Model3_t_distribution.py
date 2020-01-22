import numpy as np
from numpy.linalg import inv
from scipy.special import gamma,digamma
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import optimize
import cv2
from numpy import array


images = 1000
testing = 100

Exp = np.zeros(len_train)
Exp_log = np.zeros(len_train)
delta = np.zeros(len_train)    

def get_Exp(i_index,v,mu,sigma,X):
    D = mu.shape[0]
    term1 = np.matmul((X[i_index].reshape(-1,1)-mu).T , inv(sigma))
    term2 = np.matmul(term1, X[i_index].reshape(-1,1)-mu)[0,0]
    val = (v + D) / (v + term2)
    return val
    
def get_Exp_log(i_index,v,mu,sigma,X):
    D = mu.shape[0]
    term1 = np.matmul((X[i_index].reshape(-1,1)-mu).T , inv(sigma))
    term2 = np.matmul(term1, X[i_index].reshape(-1,1)-mu)[0,0]
    val = digamma((v+D)/2) - np.log( (v + term2)/2 )
    return val
def E_step(v,mu,sigma,X):
    for i in range(0,len_train):
        term = np.matmul( (X[i].reshape(-1,1)-mu).T , inv(sigma))
        delta[i] = np.matmul(term , (X[i].reshape(-1,1) - mu))
        E_h[i] = get_Exp(i,v,mu,sigma,X)
        E_log_h[i] = get_Exp_log(i,v,mu,sigma,X)
    return [delta, E_h, E_log_h]
def perform_EM_round(v,mu,sigma,X):
    D = mu.shape[0]
    global delta, E_h, E_log_h    
    print("E-step")
    [delta, E_h, E_log_h] = E_step(v,mu,sigma,X)
    
    'Updating Mean'            
    temp_mean = np.zeros((D,1))
    denom = 0
    for i in range(0,len_train):
        temp_mean = temp_mean + E_h[i]*X[i].reshape(-1,1)
        denom = denom + E_h[i]
    mu = temp_mean/denom    
        
    print("Updating Variance")
    num = np.zeros((D,D))
    for i in range(0,len_train):
        prod = np.matmul( (X[i].reshape(-1,1) - mu) , (X[i].reshape(-1,1) - mu).T )
        num = num + E_h[i]*prod
    sigma = num/denom
    sigma = np.diag( np.diag(sigma) )        

    print("Finding argmin v")
    v = optimize.fmin(tCost,v)          
     
    return [v[0], mu, sigma] 
def get_MC(X):
    meanX = np.mean(X, axis=0)
    covar = np.zeros((100, 100), dtype='float64')
    np.fill_diagonal(covar, np.cov(X, rowvar=False).diagonal())
    return [meanX, covar]

gc.collect()

I=1000
D=100
imageVector = np.empty((1000,100))
imageVector1 = np.empty((1000,100))
imageVector2 = np.empty((1000,100))
imageVector3 = np.empty((100,100))
imageVector4 = np.empty((100,100))
test = np.empty(100)

meanVector = np.empty(100)
for i in range(I):
	img = cv2.imread("/Users/binoythomas/Desktop/untitled folder/"+str(i+1)+str(1)+".jpg")
	im_resized = cv2.resize(img, (10, 10), interpolation=cv2.INTER_LINEAR)
	image = cv2.cvtColor(im_resized, cv2.COLOR_RGB2GRAY)
	imageVector1[i][:] = image.flatten()


for i in range(I):
	img = cv2.imread("/Users/binoythomas/Desktop/untitled folder 2/"+str(i+1)+str(1)+".jpg")
	im_resized = cv2.resize(img, (10, 10), interpolation=cv2.INTER_LINEAR)
	image = cv2.cvtColor(im_resized, cv2.COLOR_RGB2GRAY)
	imageVector2[i][:] = image.flatten()

for i in range(100):
	img = cv2.imread("/Users/binoythomas/Desktop/untitled folder 3/"+str(i+1)+str(1)+".jpg")
	im_resized = cv2.resize(img, (10, 10), interpolation=cv2.INTER_LINEAR)
	image = cv2.cvtColor(im_resized, cv2.COLOR_RGB2GRAY)
	imageVector3[i][:] = image.flatten()

for i in range(100):
	img = cv2.imread("/Users/binoythomas/Desktop/untitled folder 4/"+str(i+1)+str(1)+".jpg")
	im_resized = cv2.resize(img, (10, 10), interpolation=cv2.INTER_LINEAR)
	image = cv2.cvtColor(im_resized, cv2.COLOR_RGB2GRAY)
	imageVector4[i][:] = image.flatten()

[mean_face, covar_face] = get_MC(imageVector1)
[mean_nonface, covar_nonface] = get_MC(imageVector2)


for i in range(0, n_iter):
    print(i)
    [v_face, mu_face, sigma_face] = perform_EM_round(v_face,mu_face.reshape(-1,1), sigma_face, imageVector1)
    print("v_face = ", v_face)


plt.imshow(np.uint8(mu_face.reshape(10,10)))
plt.show()
plt.imshow(np.uint8((np.diag(sigma_nonface))*255/(np.diag(sigma_nonface)).max()).reshape(10,10))
#plt.imshow(np.uint8(mu1.reshape(10,10)))
#plt.imshow(np.uint8(mu_nonface.reshape(10,10)))
plt.show()

mu_face1=np.empty(100)
print(np.shape(mu_face))
for i in range(100):
    mu_face1[i]=mu_face[i]
print(np.shape(mu_face1)) 
mu_nonface1=np.empty(100)
print(np.shape(mu_nonface))
for i in range(100):
    mu_nonface1[i]=mu_nonface[i]
print(np.shape(mu_nonface1)) 

true_positive=np.zeros((101,1))
true_negative=np.zeros((101,1))
false_positive=np.zeros((101,1))
false_negative=np.zeros((101,1))

face_prob_matrix = np.empty((100))
non_face_prob_matrix = np.empty((100))

true_positive_rate=np.zeros((101,1))
false_postive_rate=np.zeros((101,1))
for count in range(101):
	
	for n in range(100):
		prob_1 = multivariate_normal.pdf(imageVector3[n][:], mu_face1,sigma_face)
		prob_2 = multivariate_normal.pdf(imageVector3[n][:], mu_nonface1,sigma_nonface)
		face_prob_matrix[n] = prob_1/(prob_1+prob_2)
		if face_prob_matrix[n] > (count*0.01):
			true_positive[count] +=1
		else:
			false_negative[count] +=1 

	
	for n in range(100):
		prob_1 = multivariate_normal.pdf(imageVector4[n][:], mu_face1,sigma_face)
		prob_2 = multivariate_normal.pdf(imageVector4[n][:], mu_nonface1,sigma_nonface)
		non_face_prob_matrix[n] = prob_2/(prob_1+prob_2)
		if non_face_prob_matrix[n] > (count*0.01):
			true_negative[count] +=1 
		else:
			false_positive[count] +=1
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
plt.title("ROC Curve t-distribution model")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()
