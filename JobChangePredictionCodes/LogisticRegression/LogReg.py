import numpy as np
import pandas as pd
#Add one extra column for the bias.
def add_bias(X):
    X = pd.concat([pd.Series(1, index = X.index, name = '00'), X], axis=1)
    return X

def hypothesis(X, theta):
    z = np.dot(theta, X.T)
    return 1/(1+np.exp(-(z))) - 0.0000001

def cost(X, y, theta, w):
    y1 = hypothesis(X, theta)
    if len(w) ==0:
        return -(1/len(X)) * np.sum(y*np.log(y1) + (1-y)*np.log(1-y1))
    else:
        return -(1/len(X)) * np.sum(y*np.log(y1)*w[0] + (1-y)*np.log(1-y1)*w[1])

def gradient_descent(X, y, theta, alpha, max_epochs, tol, w):
    m =len(X)
    J = [cost(X, y, theta, w)]
    j = 100000
    epoch=0
    while (epoch<max_epochs):
        if epoch%50 ==0:
            print("Epoch:",epoch)
            if abs(j) < tol:
                print("Converged, stop at epoch:", epoch)
                break
        h = hypothesis(X, theta)
        if len(w) ==0:
            for i in range(0, len(X.columns)):
                theta[i] -= (alpha/m) * np.sum((h-y)*X.iloc[:, i])
        else:
            for i in range(0, len(X.columns)):
                theta[i] -= (alpha/m) * np.sum((-y*(1-h)*w[1] + h*(1-y)* w[0])*X.iloc[:, i])
        j = cost(X, y, theta, w)
        J.append(j)
        epoch+=1
    return J, theta

def fit (X, y,  alpha, max_epochs,tol, theta = None, balanced =False):
    w = []
    if balanced:
        w =  (len(y)/(2* np.bincount(np.array(y, dtype = 'int64'))))
    if theta == None:
        theta = [0.5] * len(X.columns)
        J, theta = gradient_descent(X, y, theta, alpha, max_epochs, tol, w)
    else:
        J, theta = gradient_descent(X, y, theta, alpha, max_epochs, tol, w)
    return J, theta

def predict(X, y, theta, threshold = 0.5):
    h = hypothesis(X, theta)
    for i in range(len(h)):
        h[i]=1 if h[i]>=threshold else 0
    y = list(y)
    acc = np.sum([y[i] == h[i] for i in range(len(y))])/len(y)
    return h, acc

def recall_precision_f1(h, y):
    tp = np.sum([(y[i] == h[i] and h[i] ==1) for i in range(len(y))])
    fp = np.sum([(y[i] != h[i] and h[i] ==1) for i in range(len(y))])
    fn = np.sum([(y[i] != h[i] and h[i] ==0) for i in range(len(y))])
    recall = tp/(tp+ fn)
    precision = tp/(tp+ fp)
    f1 = 2*recall*precision/(recall+ precision)
    return recall, precision, f1