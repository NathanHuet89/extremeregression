
########Python Notebook for "On Regression in Extreme Regions################


##############################
##########Packages############
##############################

from sklearn.preprocessing import StandardScaler
import scipy.stats as stat
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

##############################
######Simulated datasets######
##############################

######Multivariate symmetric logistic Model
####see "Simulating Multivariate Extreme Value Distributions of Logistic Type", 2003, A. Stephenson for more details

def PS(n,alpha,seed):
    """
    Generate positive stable random variables.

    Inputs:
    - n : sample size
    - alpha : dependence parameter
    - seed : random seed

    Output:
    - Sample of positive stable random variables
    """
    U=stat.uniform.rvs(0,np.pi,size=n,random_state = seed)
    W=stat.expon.rvs(size=n,random_state = seed)
    Term_1=(np.sin((1-alpha)*U)/W)**((1-alpha)/alpha)
    Term_2=np.sin(alpha*U)/((np.sin(U))**(1/alpha))
    return Term_1*Term_2



def gen_log(n,d,alpha,Hill_index,seed):
    """
    Generate multivariate logistic random variables.

    Inputs:
    - n : sample size
    - d : dimension
    - alpha : dependence parameter
    - Hill_index : Hill index
    - seed : random seed

    Output:
    - Sample of multivariate logistic random variables
    """
    W=stat.expon.rvs(size=(n,d),random_state = seed)
    S=PS(n,alpha,seed)
    Log=np.zeros((n,d))
    for i in range(d):
        Log[:,i]=(S/W[:,i])**alpha
    return Log**(1/Hill_index)

######Regression functions

##Additive noise model
def f1_add(R):
    return 1+1/R.reshape(-1,1)

def f2_add(A,beta):
    return np.dot(A,beta)


def f_add(X,beta):
    """
    Function tilde(g) for the additive noise model

    Inputs:
    - X : Numpy array
    - beta : vector beta

    Output:
    -  tilde(g)_0(X)
    """
    rad = np.linalg.norm(X,axis=1)
    ang = X/rad[:, np.newaxis]
    return f1_add(rad)*f2_add(ang,beta)

##Multiplicative noise model
def f1_mult(R):
    return np.cos(1/R.reshape(-1,1))

def f2_mult(A,rad):
    s=np.zeros(len(A))
    B=A-1/rad.reshape(-1,1)**2
    for i in range(len(B[0])):
        if i%2==0:
            s=+ B[:,i]*np.sin(np.pi * (B[:,i+1]))
    return s.reshape(-1,1)

def f_mult(X):
    """
    Function tilde(g)_1 for the multiplicative noise model

    Inputs:
    - X : Numpy array

    Output:
    -  tilde(g)(X)
    """
    rad = np.linalg.norm(X,axis=1)
    ang = X/rad[:, np.newaxis]
    return f1_mult(rad)*f2_mult(ang,rad)


####Function to compute values of Table 1 for the additive noise model
def comparison_add(n_train,n_test,d,alpha,Hill_index,k_train,k_test,epoch,seed=0):
    """
    Compute mean square errors for OLS, SVR and RF algorithms for the additive noise model

    Inputs:
    - n_train : sample size of the training set
    - n_test : sample size of the test set
    - d : dimension
    - alpha : dependence parameter for the multivariate logisitic model
    - Hill_index : Hill index
    - k_train : sample size of the extreme training set
    - k_test : sample size of the extreme test set
    - epoch : number of epochs
    - seed : random seed

    Outputs:
    - mse_lin : mean square errors for OLS model trained with all observations (1xe)
    - mse_SVR : mean square errors for SVR model trained with all observations (1xe)
    - mse_RF : mean square errors for RF model trained with all observations (1xe)

    - mse_th_lin : mean square errors for OLS model trained with extreme observations (1xe)
    - mse_th_SVR : mean square errors for OLS model trained with extreme observations (1xe)
    - mse_th_RF : mean square errors for OLS model trained with extreme observations (1xe)

    - mse_th_ang_lin : mean square errors for OLS model trained with extreme angles (1xe)
    - mse_th_ang_SVR : mean square errors for OLS model trained with extreme angles (1xe)
    - mse_th_ang_R : mean square errors for OLS model trained with extreme angles (1xe)
    """
    mse_th_lin=np.zeros(shape=(1,epoch))
    mse_th_ang_lin=np.zeros(shape=(1,epoch))
    mse_lin=np.zeros(shape=(1,epoch))

    mse_th_SVR=np.zeros(shape=(1,epoch))
    mse_th_ang_SVR=np.zeros(shape=(1,epoch))
    mse_SVR=np.zeros(shape=(1,epoch))

    mse_th_RF=np.zeros(shape=(1,epoch))
    mse_th_ang_RF=np.zeros(shape=(1,epoch))
    mse_RF=np.zeros(shape=(1,epoch))

    for e in range(epoch):
        print(e)

        #Dataset generation
        beta = stat.uniform.rvs(size=(d,1),random_state = seed + e)
        noise_train = stat.truncnorm.rvs(-1,1,loc=0,scale=0.1,size=n_train,random_state = seed + e)
        noise_test = stat.truncnorm.rvs(-1,1,loc=0,scale=0.1,size=n_test,random_state = seed + e + epoch)

        X_train = gen_log(n_train,d,alpha,Hill_index, seed = seed + e) #1st input trainset
        X_test = gen_log(n_test,d,alpha,Hill_index,seed = seed + e + epoch)

        y_train = f_add(X_train,beta).ravel()+noise_train
        y_test = f_add(X_test,beta).ravel()+noise_test

        #TRAINSET
        rad_train = np.linalg.norm(X_train,axis=1)
        index_ext_train = rad_train.argsort()[::-1][:k_train]

        X_train_th = X_train[index_ext_train] #2nd input trainset
        X_train_th_ang = X_train_th/rad_train[index_ext_train].reshape(-1,1) #3rd input trainset

        y_train_th = y_train[index_ext_train] #output trainset


        #TESTSET
        rad_test = np.linalg.norm(X_test,axis=1)
        index_ext_test = rad_test.argsort()[::-1][:k_test]

        X_test_th = X_test[index_ext_test] #1st output testset
        X_test_th_ang = X_test_th/rad_test[index_ext_test].reshape(-1,1) #2nd output testset

        y_test_th = y_test[index_ext_test] #output testset

        #OLS

        lin = LinearRegression()
        lin.fit(X_train, y_train)
        y_pred_test = lin.predict(X_test_th)
        mse_lin[0][e] = mean_squared_error(y_test_th,y_pred_test)

        th_lin = LinearRegression()
        th_lin.fit(X_train_th, y_train_th)
        y_pred_test_th_lin = th_lin.predict(X_test_th)
        mse_th_lin[0][e] = mean_squared_error(y_test_th,y_pred_test_th_lin)

        th_ang_lin = LinearRegression()
        th_ang_lin.fit(X_train_th_ang, y_train_th)
        y_pred_test_th_ang_lin = th_ang_lin.predict(X_test_th_ang)
        mse_th_ang_lin[0][e] = mean_squared_error(y_test_th,y_pred_test_th_ang_lin)

        #SVR

        SVreg = SVR(kernel='rbf')
        SVreg.fit(X_train, y_train)
        y_pred_test = SVreg.predict(X_test_th)
        mse_SVR[0][e] = mean_squared_error(y_test_th,y_pred_test)

        th_SVreg = SVR(kernel='rbf')
        th_SVreg.fit(X_train_th, y_train_th)
        y_pred_test_th_SVR = th_SVreg.predict(X_test_th)
        mse_th_SVR[0][e] = mean_squared_error(y_test_th,y_pred_test_th_SVR)

        th_ang_SVreg = SVR(kernel='rbf')
        th_ang_SVreg.fit(X_train_th_ang, y_train_th)
        y_pred_test_th_ang_SVR = th_ang_SVreg.predict(X_test_th_ang)
        mse_th_ang_SVR[0][e] = mean_squared_error(y_test_th,y_pred_test_th_ang_SVR)

        #RandomForest

        RF = RandomForestRegressor()
        RF.fit(X_train, y_train)
        y_pred_test = RF.predict(X_test_th)
        mse_RF[0][e] = mean_squared_error(y_test_th,y_pred_test)

        th_RF = RandomForestRegressor()
        th_RF.fit(X_train_th, y_train_th)
        y_pred_test_th_RF = th_RF.predict(X_test_th)
        mse_th_RF[0][e] = mean_squared_error(y_test_th,y_pred_test_th_RF)

        th_ang_RF = RandomForestRegressor()
        th_ang_RF.fit(X_train_th_ang, y_train_th)
        y_pred_test_th_ang_RF = th_ang_RF.predict(X_test_th_ang)
        mse_th_ang_RF[0][e] = mean_squared_error(y_test_th,y_pred_test_th_ang_RF)

    return(mse_lin,mse_SVR,mse_RF,mse_th_lin,mse_th_SVR,mse_th_RF,mse_th_ang_lin,mse_th_ang_SVR,mse_th_ang_RF)

####Table 1 values for additive noise model
#Parameters for the additive noise model:
n_train,n_test,d,alpha,Hill_index = 10000,100000,7,1,1
k_test=int(n_test/10)
k_train=int(n_train/10)
e=10
seed = 0

#MSE computation:
mse_lin,mse_SVR,mse_RF,mse_th_lin,mse_th_SVR,mse_th_RF,mse_th_ang_lin,mse_th_ang_SVR,mse_th_ang_RF = comparison_add(n_train,n_test,d,alpha,Hill_index,k_train,k_test,e,seed=0)

mse = [mse_lin[0],mse_SVR[0],mse_RF[0]]
mse_th = [mse_th_lin[0],mse_th_SVR[0],mse_th_RF[0]]
mse_th_ang = [mse_th_ang_lin[0],mse_th_ang_SVR[0],mse_th_ang_RF[0]]

#MSE mean:
mean = np.mean(mse,axis=1)
mean_th = np.mean(mse_th,axis=1)
mean_th_ang = np.mean(mse_th_ang,axis=1)

#MSE standard deviation:
std = np.sqrt(np.var(mse,axis=1))
std_th = np.sqrt(np.var(mse_th,axis=1))
std_th_ang = np.sqrt(np.var(mse_th_ang,axis=1))

###Function to compute values of Table 1 for the multiplicative noise model
def comparison_mult(n_train,n_test,d,alpha,Hill_index,k_train,k_test,epoch,seed=0):
    """
    Compute mean square errors for OLS, SVR and RF algorithms for the multiplicative noise model

    Inputs:
    - n_train : sample size of the training set
    - n_test : sample size of the test set
    - d : dimension
    - alpha : dependence parameter for the multivariate logisitic model
    - Hill_index : Hill index
    - k_train : sample size of the extreme training set
    - k_test : sample size of the extreme test set
    - epoch : number of epochs
    - seed : random seed

    Outputs:
    - mse_lin : mean square errors for OLS model trained with all observations (1 x epoch)
    - mse_SVR : mean square errors for SVR model trained with all observations (1 x epoch)
    - mse_RF : mean square errors for RF model trained with all observations (1 x epoch)

    - mse_th_lin : mean square errors for OLS model trained with extreme observations (1 x epoch)
    - mse_th_SVR : mean square errors for OLS model trained with extreme observations (1 x epoch)
    - mse_th_RF : mean square errors for OLS model trained with extreme observations (1 x epoch)

    - mse_th_ang_lin : mean square errors for OLS model trained with extreme angles (1 x epoch)
    - mse_th_ang_SVR : mean square errors for OLS model trained with extreme angles (1 x epoch)
    - mse_th_ang_R : mean square errors for OLS model trained with extreme angles (1 x epoch)
    """
    mse_th_lin=np.zeros(shape=(1,epoch))
    mse_th_ang_lin=np.zeros(shape=(1,epoch))
    mse_lin=np.zeros(shape=(1,epoch))


    mse_th_SVR=np.zeros(shape=(1,epoch))
    mse_th_ang_SVR=np.zeros(shape=(1,epoch))
    mse_SVR=np.zeros(shape=(1,epoch))

    mse_th_RF=np.zeros(shape=(1,epoch))
    mse_th_ang_RF=np.zeros(shape=(1,epoch))
    mse_RF=np.zeros(shape=(1,epoch))

    for e in range(epoch):
        print(e)

        #Dataset generation
        noise_train = stat.truncnorm.rvs(0,2,loc=1,scale=0.1,size=n_train,random_state = seed + e)
        noise_test = stat.truncnorm.rvs(0,2,loc=1,scale=0.1,size=n_test,random_state = seed + e + epoch)

        X_train = gen_log(n_train,d,alpha,Hill_index, seed = seed + e) #1st trainset
        X_test = gen_log(n_test,d,alpha,Hill_index,seed = seed + e + epoch)

        y_train = f_mult(X_train).ravel()*noise_train
        y_test = f_mult(X_test).ravel()*noise_test

        #TRAINSET
        rad_train = np.linalg.norm(X_train,axis=1)
        index_ext_train = rad_train.argsort()[::-1][:k_train]

        X_train_th = X_train[index_ext_train] #2nd trainset
        X_train_th_ang = X_train_th/rad_train[index_ext_train].reshape(-1,1) #3rd trainset

        y_train_th = y_train[index_ext_train] #output trainset

        #TESTSET
        rad_test = np.linalg.norm(X_test,axis=1)
        index_ext_test = rad_test.argsort()[::-1][:k_test]

        X_test_th = X_test[index_ext_test] #1st testset
        X_test_th_ang = X_test_th/rad_test[index_ext_test].reshape(-1,1) #2nd testset

        y_test_th = y_test[index_ext_test] #output testset

        #OLS

        lin = LinearRegression()
        lin.fit(X_train, y_train)
        y_pred_test = lin.predict(X_test_th)
        mse_lin[0][e] = mean_squared_error(y_test_th,y_pred_test)

        th_lin = LinearRegression()
        th_lin.fit(X_train_th, y_train_th)
        y_pred_test_th_lin = th_lin.predict(X_test_th)
        mse_th_lin[0][e] = mean_squared_error(y_test_th,y_pred_test_th_lin)

        th_ang_lin = LinearRegression()
        th_ang_lin.fit(X_train_th_ang, y_train_th)
        y_pred_test_th_ang_lin = th_ang_lin.predict(X_test_th_ang)
        mse_th_ang_lin[0][e] = mean_squared_error(y_test_th,y_pred_test_th_ang_lin)

        #SVR

        SVreg = SVR(kernel='rbf')
        SVreg.fit(X_train, y_train)
        y_pred_test = SVreg.predict(X_test_th)
        mse_SVR[0][e] = mean_squared_error(y_test_th,y_pred_test)

        th_SVreg = SVR(kernel='rbf')
        th_SVreg.fit(X_train_th, y_train_th)
        y_pred_test_th_SVR = th_SVreg.predict(X_test_th)
        mse_th_SVR[0][e] = mean_squared_error(y_test_th,y_pred_test_th_SVR)

        th_ang_SVreg = SVR(kernel='rbf')
        th_ang_SVreg.fit(X_train_th_ang, y_train_th)
        y_pred_test_th_ang_SVR = th_ang_SVreg.predict(X_test_th_ang)
        mse_th_ang_SVR[0][e] = mean_squared_error(y_test_th,y_pred_test_th_ang_SVR)

        #RandomForest

        RF = RandomForestRegressor()
        RF.fit(X_train, y_train)
        y_pred_test = RF.predict(X_test_th)
        mse_RF[0][e] = mean_squared_error(y_test_th,y_pred_test)

        th_RF = RandomForestRegressor()
        th_RF.fit(X_train_th, y_train_th)
        y_pred_test_th_RF = th_RF.predict(X_test_th)
        mse_th_RF[0][e] = mean_squared_error(y_test_th,y_pred_test_th_RF)

        th_ang_RF = RandomForestRegressor()
        th_ang_RF.fit(X_train_th_ang, y_train_th)
        y_pred_test_th_ang_RF = th_ang_RF.predict(X_test_th_ang)
        mse_th_ang_RF[0][e] = mean_squared_error(y_test_th,y_pred_test_th_ang_RF)

    return(mse_lin,mse_SVR,mse_RF,mse_th_lin,mse_th_SVR,mse_th_RF,mse_th_ang_lin,mse_th_ang_SVR,mse_th_ang_RF)

####Table 1 values for multiplicative noise model
#Parameters for the additive noise model:
n_train,n_test,d,alpha,Hill_index = 10000,100000,14,0.7,3
k_test=int(n_test/10)
k_train=int(n_train/10)

e=10

seed = 0

#MSE computation:
mse_lin,mse_SVR,mse_RF,mse_th_lin,mse_th_SVR,mse_th_RF,mse_th_ang_lin,mse_th_ang_SVR,mse_th_ang_RF = comparison_mult(n_train,n_test,d,alpha,Hill_index,k_train,k_test,e,seed=0)

mse = [mse_lin[0],mse_SVR[0],mse_RF[0]]
mse_th = [mse_th_lin[0],mse_th_SVR[0],mse_th_RF[0]]
mse_th_ang = [mse_th_ang_lin[0],mse_th_ang_SVR[0],mse_th_ang_RF[0]]

#MSE mean:
mean = np.mean(mse,axis=1)
mean_th = np.mean(mse_th,axis=1)
mean_th_ang = np.mean(mse_th_ang,axis=1)

#MSE standard deviation:
std = np.sqrt(np.var(mse,axis=1))
std_th = np.sqrt(np.var(mse_th,axis=1))
std_th_ang = np.sqrt(np.var(mse_th_ang,axis=1))

##Fonction to compute the Gini and permutation values for the additive noise model
def signif_k_constant_add(h,n_max,k,d,alpha,Hill_index,epoch,seed=0):
    """
    Compute Gini and permutation importances of the radial variable for the RF model of the additive noise model

    Inputs:
    - h : number of steps
    - n_max : maximal sample size
    - k : extreme sample size
    - d : dimension
    - alpha : dependence parameter for the multivariate logisitic model
    - Hill_index : Hill index
    - epoch : number of epochs
    - seed : random seed

    Outputs:
    - importance_rayon_RF : Gini importance of the radial variable (epoch x h)
    - permut_importance_RF : permutation importance of the radial variable (epoch x h)
    """
    importance_rayon_RF = np.zeros((epoch,h))
    permut_importance_RF = np.zeros((epoch,h))

    for e in range(epoch):
        print(e)

        #Dataset generation of size n_max
        beta = stat.uniform.rvs(size=(d,1),random_state=seed+e)
        beta_norm = beta / np.linalg.norm(beta)

        noise = stat.truncnorm.rvs(-1,1,loc=0,scale=0.1,size=n_max,random_state=seed+e)
        X_tot = gen_log(n_max,d,alpha,Hill_index,seed)
        y_tot = f_add(X_tot,beta_norm).ravel()+noise

        for n in range(n_max//h,n_max+1,n_max//h):
            #Dataset extraction of size n
            X = X_tot[:n]
            y = y_tot[:n]


            rad = np.linalg.norm(X,axis=1)
            X_ang = X/rad.reshape(-1,1)
            X_ang_rad = np.c_[ X_ang, rad ] #the radial variable is the last one
            index_ext = rad.argsort()[::-1][:k-1]

            index_ext_train = index_ext[:k-1]
            X_train = X_ang_rad[index_ext_train]
            y_train = y[index_ext_train]

            #RF importances
            RF = RandomForestRegressor()
            RF.fit(X_train, y_train.ravel())
            importance_rayon_RF[e][n//(n_max//h)-1] = RF.feature_importances_[-1]
            permut_RF = permutation_importance(RF, X_train, y_train.ravel())
            permut_importance_RF[e][n//(n_max//h)-1] = permut_RF.importances_mean[-1]

    return(importance_rayon_RF,permut_importance_RF)

##Fonction to compute the Gini and permutation values for the additive noise model
def signif_k_constant_mult(h,n_max,k,d,alpha,Hill_index,epoch,seed=0):
    """
    Compute Gini and permutation importances of the radial variable for the RF model of the multiplicative noise model

    Inputs:
    - h : number of steps
    - n_max : maximal sample size
    - k : extreme sample size
    - d : dimension
    - alpha : dependence parameter for the multivariate logisitic model
    - Hill_index : Hill index
    - epoch : number of epochs
    - seed : random seed

    Outputs:
    - importance_rayon_RF : Gini importance of the radial variable (epoch x h)
    - permut_importance_RF : permutation importance of the radial variable (epoch x h)
    """
    importance_rayon_RF = np.zeros((epoch,h))
    permut_importance_RF = np.zeros((epoch,h))

    for e in range(epoch):
        print(e)

        #Dataset generation of size n_max
        noise = stat.truncnorm.rvs(0,2,loc=1,scale=0.1,size=n_max,random_state=seed+e)
        X_tot = gen_log(n_max,d,alpha,Hill_index,seed)
        y_tot = f_mult(X_tot).ravel()*noise

        for n in range(n_max//h,n_max+1,n_max//h):

            #Dataset extraction of size n
            X = X_tot[:n]
            y = y_tot[:n]

            rad = np.linalg.norm(X,axis=1)
            X_ang = X/rad.reshape(-1,1)
            X_ang_rad = np.c_[ X_ang, rad ] #the radial variable is the last one
            index_ext = rad.argsort()[::-1][:k-1]

            index_ext_train = index_ext[:k-1]
            X_train = X_ang_rad[index_ext_train]
            y_train = y[index_ext_train]

            #RF importances
            RF = RandomForestRegressor()
            RF.fit(X_train, y_train.ravel())
            importance_rayon_RF[e][n//(n_max//h)-1] = RF.feature_importances_[-1]
            permut_RF = permutation_importance(RF, X_train, y_train.ravel())
            permut_importance_RF[e][n//(n_max//h)-1] = permut_RF.importances_mean[-1]

    return(importance_rayon_RF,permut_importance_RF)

####Plot Figure 1
##Additive noise model
#Parameters:
h,k,n_max=10,1000,10000
d,alpha,Hill_index = 7,1,1
seed=0

epoch=10



#computing of importances:
importance_rayon_RF,permut_importance_RF=signif_k_constant_add(h,n_max,k,d,alpha,Hill_index,epoch,seed)

#plot figure 1:
mean_importance_rayon_RF = np.mean(importance_rayon_RF,axis=0)
mean_permut_importance_RF = np.mean(permut_importance_RF,axis=0)

permut95=np.percentile(permut_importance_RF, 95, axis=0)
permut5=np.percentile(permut_importance_RF, 5, axis=0)

gini95 = np.percentile(importance_rayon_RF, 95, axis=0)
gini5 = np.percentile(importance_rayon_RF, 5, axis=0)

x=np.arange(n_max//h,n_max+1,n_max//h)

plt.plot(x,mean_importance_rayon_RF,color='black')
plt.plot(x,gini95,ls='--',color='black')
plt.plot(x,gini5,ls='--',color='black')

plt.plot(x,mean_permut_importance_RF,color='black',alpha=0.4)
plt.plot(x,permut95,ls='--',alpha=0.4,color='black')
plt.plot(x,permut5,ls='--',alpha=0.4,color='black')

plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.ylabel('Importance score',fontsize=30)
plt.xlabel('n',fontsize=30)
plt.show()

##Multiplicative noise model
#Parameters:
h,k,n_max=10,1000,10000
d,alpha,Hill_index = 14,0.7,3
seed=0

epoch=10



#importance computation:
importance_rayon_RF,permut_importance_RF=signif_k_constant_mult(h,n_max,k,d,alpha,Hill_index,epoch,seed)

#plot figure 1:
mean_importance_rayon_RF = np.mean(importance_rayon_RF,axis=0)
mean_permut_importance_RF = np.mean(permut_importance_RF,axis=0)

permut95=np.percentile(permut_importance_RF, 95, axis=0)
permut5=np.percentile(permut_importance_RF, 5, axis=0)

gini95 = np.percentile(importance_rayon_RF, 95, axis=0)
gini5 = np.percentile(importance_rayon_RF, 5, axis=0)

x=np.arange(n_max//h,n_max+1,n_max//h)

plt.plot(x,mean_importance_rayon_RF,color='black')
plt.plot(x,gini95,ls='--',color='black')
plt.plot(x,gini5,ls='--',color='black')

plt.plot(x,mean_permut_importance_RF,color='black',alpha=0.4)
plt.plot(x,permut95,ls='--',alpha=0.4,color='black')
plt.plot(x,permut5,ls='--',alpha=0.4,color='black')

plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.ylabel('Importance score',fontsize=30)
plt.xlabel('n',fontsize=30)
plt.show()

#########################
######Real Dataset#######
#########################


##data importation
df = pd.read_csv('49_Industry_Portfolios_Daily.csv', low_memory = False)
df_copy = df.copy()

df_copy = df_copy.apply(pd.to_numeric, errors='coerce')


#dates are removed
df_copy = df_copy.drop('Unnamed: 0', axis=1)

##Functions for Hill plot
def get_moments_estimates(ordered_data):
    """
    Compute the first moment Hill estimator.

    Input:
    - ordered_data : ordered data

    Output:
    - Hill estimator
    """
    logs_1 = np.log(ordered_data)
    logs_1_cumsum = np.cumsum(logs_1[:-1])
    k_vector = np.arange(1, len(ordered_data))
    M1 = (1./k_vector)*logs_1_cumsum - logs_1[1:]
    return M1

##Hill plot of the 2-norm of the data

#ordered radial data
rad = np.linalg.norm(df_copy,axis=1)
rad[::-1].sort(axis=0)

#Hill estimate computation
RV_index = 1/get_moments_estimates(rad)

#plot figure 2:
plt.plot(RV_index[:4000],color='black')
plt.xticks(fontsize=16)
plt.yticks([3,3.2,3.5,4,4.5,5,5.5,6],fontsize=16)
plt.axhline(y=3.2, color='black', linestyle='--', linewidth=2,alpha=0.5)
plt.ylabel(r'$\hat{\alpha}$',fontsize=20)
plt.xlabel('k',fontsize=20)
plt.show()
##Fonction to compute values of Table 2
def comparison_real(X,y,k_train,k_test,epoch,seed=0):
    """
    Compute mean square errors for OLS, SVR and RF algorithms for the real dataset

    Inputs:
    - X : input array
    - y : output array
    - k_train : sample size of the extreme training set
    - k_test : sample size of the extreme test set
    - epoch : number of epoch
    - seed : random seed

    Outputs:
    - mse_lin : mean square errors for OLS model trained with all observations (1xe)
    - mse_SVR : mean square errors for SVR model trained with all observations (1xe)
    - mse_RF : mean square errors for RF model trained with all observations (1xe)

    - mse_th_lin : mean square errors for OLS model trained with extreme observations (1xe)
    - mse_th_SVR : mean square errors for OLS model trained with extreme observations (1xe)
    - mse_th_RF : mean square errors for OLS model trained with extreme observations (1xe)

    - mse_th_ang_lin : mean square errors for OLS model trained with extreme angles (1xe)
    - mse_th_ang_SVR : mean square errors for OLS model trained with extreme angles (1xe)
    - mse_th_ang_R : mean square errors for OLS model trained with extreme angles (1xe)
    """

    mse_th_lin=np.zeros(shape=(1,epoch))
    mse_th_ang_lin=np.zeros(shape=(1,epoch))
    mse_lin=np.zeros(shape=(1,epoch))

    mse_th_SVR=np.zeros(shape=(1,epoch))
    mse_th_ang_SVR=np.zeros(shape=(1,epoch))
    mse_SVR=np.zeros(shape=(1,epoch))

    mse_th_RF=np.zeros(shape=(1,epoch))
    mse_th_ang_RF=np.zeros(shape=(1,epoch))
    mse_RF=np.zeros(shape=(1,epoch))

    for e in range(epoch):
        print(e)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state= seed + e)

        #TRAINSET
        rad_train = np.linalg.norm(X_train,axis=1)
        index_ext_train = rad_train.argsort()[::-1][:k_train]

        X_train_th = X_train[index_ext_train] #2ème trainset
        X_train_th_ang = X_train_th/rad_train[index_ext_train].reshape(-1,1) #3ème trainset

        y_train_th = y_train[index_ext_train] #1er output trainset
        y_train_th_ang =y_train_th /np.sqrt(rad_train[index_ext_train]**2+y_train_th**2) #2eme output trainset (angular)

        #TESTSET
        rad_test = np.linalg.norm(X_test,axis=1)
        index_ext_test = rad_test.argsort()[::-1][:k_test]

        X_test_th = X_test[index_ext_test] #1er testset
        X_test_th_ang = X_test_th/rad_test[index_ext_test].reshape(-1,1) #2ème testset

        y_test_th = y_test[index_ext_test]

        #variable selection for the OLS algorithm
        correlations = np.abs(np.corrcoef(X, y, rowvar=False)[:,-1])

        top_columns_indices = np.argsort(correlations)[::-1][1:11]

        X_train_reduc = X_train[:, top_columns_indices]
        X_test_reduc = X_test[:, top_columns_indices]

        #train and test sets for the OLS algorithm
        #TRAINSET reduc
        rad_train_reduc = np.linalg.norm(X_train_reduc,axis=1)
        index_ext_train_reduc = rad_train_reduc.argsort()[::-1][:k_train]

        X_train_reduc_th = X_train_reduc[index_ext_train_reduc] #2ème trainset
        X_train_reduc_th_ang = X_train_reduc_th/rad_train_reduc[index_ext_train_reduc].reshape(-1,1) #3ème trainset

        y_train_reduc_th = y_train[index_ext_train_reduc] #1er output trainset
        y_train_reduc_th_ang =y_train_reduc_th /np.sqrt(rad_train_reduc[index_ext_train_reduc]**2+y_train_reduc_th**2) #2eme output trainset (angular)

        #TESTSET reduc
        rad_test_reduc = np.linalg.norm(X_test_reduc,axis=1)
        index_ext_test_reduc = rad_test_reduc.argsort()[::-1][:k_test]

        X_test_reduc_th = X_test_reduc[index_ext_test_reduc] #1er testset
        X_test_reduc_th_ang = X_test_reduc_th/rad_test_reduc[index_ext_test_reduc].reshape(-1,1) #2ème testset

        y_test_reduc_th = y_test[index_ext_test_reduc]


        #lin_reg

        lin = LinearRegression()
        lin.fit(X_train_reduc, y_train)
        y_pred_test = lin.predict(X_test_reduc_th)
        mse_lin[0][e] = mean_squared_error(y_test_reduc_th,y_pred_test)

        th_lin = LinearRegression()
        th_lin.fit(X_train_reduc_th, y_train_reduc_th)
        y_pred_test_th_lin = th_lin.predict(X_test_reduc_th)
        mse_th_lin[0][e] = mean_squared_error(y_test_reduc_th,y_pred_test_th_lin)

        th_ang_lin = LinearRegression()
        th_ang_lin.fit(X_train_reduc_th_ang, y_train_reduc_th_ang)
        y_pred_test_th_ang_lin = th_ang_lin.predict(X_test_reduc_th_ang)
        y_pred_test_th_ang_lin_ori = y_pred_test_th_ang_lin * rad_test_reduc[index_ext_test_reduc] /(1 - y_pred_test_th_ang_lin**2)**(1/2)
        mse_th_ang_lin[0][e]=mean_squared_error(y_test_reduc_th,y_pred_test_th_ang_lin_ori)


        #SVR
        SVreg = SVR(kernel='rbf')
        SVreg.fit(X_train, y_train)
        y_pred_test = SVreg.predict(X_test_th)
        mse_SVR[0][e] = mean_squared_error(y_test_th,y_pred_test)

        th_SVreg = SVR(kernel='rbf')
        th_SVreg.fit(X_train_th, y_train_th)
        y_pred_test_th_SVR = th_SVreg.predict(X_test_th)
        mse_th_SVR[0][e] = mean_squared_error(y_test_th,y_pred_test_th_SVR)

        th_ang_SVreg = SVR(kernel='rbf')
        th_ang_SVreg.fit(X_train_th_ang, y_train_th_ang)
        y_pred_test_th_ang_SVR = th_ang_SVreg.predict(X_test_th_ang)
        y_pred_test_th_ang_SVR_ori = y_pred_test_th_ang_SVR * rad_test[index_ext_test]/(1 - y_pred_test_th_ang_SVR**2)**(1/2)
        mse_th_ang_SVR[0][e] = mean_squared_error(y_test_th,y_pred_test_th_ang_SVR_ori)

        #RandomForest

        RF = RandomForestRegressor()
        RF.fit(X_train, y_train)
        y_pred_test = RF.predict(X_test_th)
        mse_RF[0][e] = mean_squared_error(y_test_th,y_pred_test)

        th_RF = RandomForestRegressor()
        th_RF.fit(X_train_th, y_train_th)
        y_pred_test_th_RF = th_RF.predict(X_test_th)
        mse_th_RF[0][e] = mean_squared_error(y_test_th,y_pred_test_th_RF)

        th_ang_RF = RandomForestRegressor()
        th_ang_RF.fit(X_train_th_ang, y_train_th_ang)
        y_pred_test_th_ang_RF = th_ang_RF.predict(X_test_th_ang)
        y_pred_test_th_ang_RF_ori = y_pred_test_th_ang_RF * rad_test[index_ext_test] /(1 - y_pred_test_th_ang_RF**2)**(1/2)
        mse_th_ang_RF[0][e] = mean_squared_error(y_test_th,y_pred_test_th_ang_RF_ori)

    return(mse_lin,mse_SVR,mse_RF,mse_th_lin,mse_th_SVR,mse_th_RF,mse_th_ang_lin,mse_th_ang_SVR,mse_th_ang_RF)


##Fonction to compute the Gini and permutation importance values for the real dataset
def signif_k_constant_real(h,X_tot,y_tot,epoch,seed=0):
    """
    Compute Gini and permutation importances of the radial variable for the RF model of the real datset

    Inputs:
    - h : number of step
    - X_tot : input array
    - y_tot : output array
    - epoch : number of epoch
    - seed : random seed

    Outputs:
    - importance_rayon_RF : Gini importance of the radial variable (epoch x h)
    - permut_importance_RF : permutation importance of the radial variable (epoch x h)
    """
    n_max=len(X_tot)
    k=int(len(X_tot)/h)


    importance_rayon_RF = np.zeros((epoch,h))
    permut_importance_RF = np.zeros((epoch,h))

    for e in range(epoch):
        print(e)
        np.random.seed(seed+e)

        #random shuffling for epoch e
        shuffled_index = np.random.permutation(np.arange(len(X_tot)))
        X_tot = X_tot[shuffled_index]
        y_tot = y_tot[shuffled_index]

        for n in range(n_max//h,n_max+1,n_max//h):
            #we consider the first n data at each iteration
            X = X_tot[:n]
            y = y_tot[:n]

            rad = np.linalg.norm(X,axis=1)
            X_ang = X/rad.reshape(-1,1)
            X_ang_rad = np.c_[ X_ang, rad ] #the radial variable is the last one

            index_ext = rad.argsort()[::-1][:k-1]

            X_th_ang_rad = X_ang_rad[index_ext] #input set

            y_th = y[index_ext]


            RF = RandomForestRegressor()
            RF.fit(X_th_ang_rad, y_th)

            #Gini importance
            importance_rayon_RF[e][n//(n_max//h)-1] = RF.feature_importances_[-1]

            #permuation importance
            permut_RF = permutation_importance(RF, X_th_ang_rad, y_th)
            permut_importance_RF[e][n//(n_max//h)-1] = permut_RF.importances_mean[-1]


    return(importance_rayon_RF,permut_importance_RF)

####Output:Agric
##Table 2 values
rad = np.linalg.norm(df_copy.to_numpy(),axis=1)

array_inputs =df_copy.drop(columns='Agric').to_numpy()


array_outputs_non_ang =  df_copy['Agric'].to_numpy()
array_outputs = array_outputs_non_ang #regression on the angular component of output variable


X=array_inputs

y=array_outputs

#parameters:
k_train = int(0.2*0.7*len(y))
k_test = int(0.1*0.3*len(y))
epoch = 10

#MSE computation
mse_lin,mse_SVR,mse_RF,mse_th_lin,mse_th_SVR,mse_th_RF,mse_th_ang_lin,mse_th_ang_SVR,mse_th_ang_RF = comparison_real(X,y,k_train,k_test,epoch,seed=0)

mse = [mse_lin[0],mse_SVR[0],mse_RF[0]]
mse_th_ang = [mse_th_ang_lin[0],mse_th_ang_SVR[0],mse_th_ang_RF[0]]
mse_th = [mse_th_lin[0],mse_th_SVR[0],mse_th_RF[0]]

#MSE mean
mean_th_ang = np.mean(mse_th_ang,axis=1)
mean_th = np.mean(mse_th,axis=1)
mean = np.mean(mse,axis=1)

#MSE standard deviation
std_th_ang = np.sqrt(np.var(mse_th_ang,axis=1))
std_th = np.sqrt(np.var(mse_th,axis=1))
std = np.sqrt(np.var(mse,axis=1))

##Figure 3 plot
#parameters:
h=10
epoch = 10

y_ang = y/rad
#importance computation
signif=signif_k_constant_real(h,X,y_ang,epoch,seed=0)


importance_rayon_RF = signif[0]
permut_importance_RF = signif[1]

mean_importance_rayon_RF = np.mean(importance_rayon_RF,axis=0)
mean_permut_importance_RF = np.mean(permut_importance_RF,axis=0)

permut95=np.percentile(permut_importance_RF, 95, axis=0)
permut5=np.percentile(permut_importance_RF, 5, axis=0)

gini95 = np.percentile(importance_rayon_RF, 95, axis=0)
gini5 = np.percentile(importance_rayon_RF, 5, axis=0)

#plot figure 3
x=np.arange(len(X)//h,len(X)+1,len(X)//h)

plt.plot(x,mean_importance_rayon_RF,color='black')
plt.plot(x,gini95,ls='--',color='black')
plt.plot(x,gini5,ls='--',color='black')

plt.plot(x,mean_permut_importance_RF,color='black',alpha=0.4)
plt.plot(x,permut95,ls='--',alpha=0.4,color='black')
plt.plot(x,permut5,ls='--',alpha=0.4,color='black')

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.ylabel('Importance score',fontsize=36)
plt.xlabel('n',fontsize=36)
plt.subplots_adjust(bottom=0.2)
plt.show()

####Output:Food
##Table 2 values

rad = np.linalg.norm(df_copy.to_numpy(),axis=1)

array_inputs =df_copy.drop(columns='Food ').to_numpy()


array_outputs_non_ang =  df_copy['Food '].to_numpy()
array_outputs = array_outputs_non_ang #regression on the angular component of output variable


X=array_inputs

y=array_outputs

#parameters:
k_train = int(0.2*0.7*len(y))
k_test = int(0.1*0.3*len(y))
epoch = 10

#MSE computation
mse_lin,mse_SVR,mse_RF,mse_th_lin,mse_th_SVR,mse_th_RF,mse_th_ang_lin,mse_th_ang_SVR,mse_th_ang_RF = comparison_real(X,y,k_train,k_test,epoch,seed=0)

mse = [mse_lin[0],mse_SVR[0],mse_RF[0]]
mse_th_ang = [mse_th_ang_lin[0],mse_th_ang_SVR[0],mse_th_ang_RF[0]]
mse_th = [mse_th_lin[0],mse_th_SVR[0],mse_th_RF[0]]

#MSE mean
mean_th_ang = np.mean(mse_th_ang,axis=1)
mean_th = np.mean(mse_th,axis=1)
mean = np.mean(mse,axis=1)

#MSE standard deviation
std_th_ang = np.sqrt(np.var(mse_th_ang,axis=1))
std_th = np.sqrt(np.var(mse_th,axis=1))
std = np.sqrt(np.var(mse,axis=1))

##Figure 3 plot
#parameters:
h=10
epoch = 10

y_ang = y/rad
#importance computation
signif=signif_k_constant_real(h,X,y_ang,epoch,seed=0)

importance_rayon_RF = signif[0]
permut_importance_RF = signif[1]

mean_importance_rayon_RF = np.mean(importance_rayon_RF,axis=0)
mean_permut_importance_RF = np.mean(permut_importance_RF,axis=0)

permut95=np.percentile(permut_importance_RF, 95, axis=0)
permut5=np.percentile(permut_importance_RF, 5, axis=0)

gini95 = np.percentile(importance_rayon_RF, 95, axis=0)
gini5 = np.percentile(importance_rayon_RF, 5, axis=0)

#figure 3 plot

x=np.arange(len(X)//h,len(X)+1,len(X)//h)

plt.plot(x,mean_importance_rayon_RF,color='black')
plt.plot(x,gini95,ls='--',color='black')
plt.plot(x,gini5,ls='--',color='black')

plt.plot(x,mean_permut_importance_RF,color='black',alpha=0.4)
plt.plot(x,permut95,ls='--',alpha=0.4,color='black')
plt.plot(x,permut5,ls='--',alpha=0.4,color='black')

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.ylabel('Importance score',fontsize=36)
plt.xlabel('n',fontsize=36)
plt.subplots_adjust(bottom=0.2)
plt.show()

####Output:Soda
##Table 2 values

rad = np.linalg.norm(df_copy.to_numpy(),axis=1)

array_inputs =df_copy.drop(columns='Soda ').to_numpy()


array_outputs_non_ang =  df_copy['Soda '].to_numpy()
array_outputs = array_outputs_non_ang #regression on the angular component of output variable


X=array_inputs

y=array_outputs

#parameters:
k_train = int(0.2*0.7*len(y))
k_test = int(0.1*0.3*len(y))
epoch = 10

#MSE computation
mse_lin,mse_SVR,,mse_RF,mse_th_lin,mse_th_SVR,mse_th_RF,mse_th_ang_lin,mse_th_ang_SVR,mse_th_ang_RF = comparison_real(X,y,k_train,k_test,epoch,seed=0)

mse = [mse_lin[0],mse_SVR[0],mse_RF[0]]
mse_th_ang = [mse_th_ang_lin[0],mse_th_ang_SVR[0],mse_th_ang_RF[0]]
mse_th = [mse_th_lin[0],mse_th_SVR[0],mse_th_RF[0]]

#MSE mean
mean_th_ang = np.mean(mse_th_ang,axis=1)
mean_th = np.mean(mse_th,axis=1)
mean = np.mean(mse,axis=1)

#MSE standard deviation
std_th_ang = np.sqrt(np.var(mse_th_ang,axis=1))
std_th = np.sqrt(np.var(mse_th,axis=1))
std = np.sqrt(np.var(mse,axis=1))

##Figure 3 plot
#parameters:
h=10
epoch = 10

y_ang = y/rad
#importance computation
signif=signif_k_constant_real(h,X,y_ang,epoch,seed=0)

importance_rayon_RF = signif[0]
permut_importance_RF = signif[1]

mean_importance_rayon_RF = np.mean(importance_rayon_RF,axis=0)
mean_permut_importance_RF = np.mean(permut_importance_RF,axis=0)

permut95=np.percentile(permut_importance_RF, 95, axis=0)
permut5=np.percentile(permut_importance_RF, 5, axis=0)

gini95 = np.percentile(importance_rayon_RF, 95, axis=0)
gini5 = np.percentile(importance_rayon_RF, 5, axis=0)

#figure 3 plot
x=np.arange(len(X)//h,len(X)+1,len(X)//h)

plt.plot(x,mean_importance_rayon_RF,color='black')
plt.plot(x,gini95,ls='--',color='black')
plt.plot(x,gini5,ls='--',color='black')

plt.plot(x,mean_permut_importance_RF,color='black',alpha=0.4)
plt.plot(x,permut95,ls='--',alpha=0.4,color='black')
plt.plot(x,permut5,ls='--',alpha=0.4,color='black')

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.ylabel('Importance score',fontsize=36)
plt.xlabel('n',fontsize=36)
plt.subplots_adjust(bottom=0.2)
plt.show()










