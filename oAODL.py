# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 11:17:08 2021

@author: yokoxue
"""

import numpy as np
from sklearn import linear_model
import time

class Online_AODL:
    def __init__(self,  lam = 0.01,Num_iteration = 1):
       self.lam, self.Num_iteration= lam, Num_iteration
    
    def data_iter(self, batch_size, Y):
        num_examples = Y.shape[1]
        indices = list(range(num_examples))
        
    # The examples are read at random, in no particular order
       # random.shuffle(indices)
        for i in range(0, num_examples, batch_size):
            batch_indices = np.array(indices[i: min(i + batch_size, num_examples)])
            yield Y[:,batch_indices]
    
    def normalize(self, D):
        n = D.shape[1]
        for i in range(n):
            d = D[:, i]
            s = np.linalg.norm(d)
            D[:, i] = d / s
        return D
    
    def Dic_Update(self, A,B,D):
         eps = 1e-9
         max_iter = 1
         n, k = D.shape
         for j in range(max_iter):
             
             for i in range(k):
                 d0 = D[:, i]
                 if A[i, i] == 0.:
                     s = 1. / eps
                 else:
                     s = 1. / A[i, i]
                 u = s * (B[:, i] - np.dot(D, A[:, i])) + d0
                 d1 = (1. / max(1., np.linalg.norm(u))) * u
                 D[:, i] = d1
         return D
                
    def Sparse_Coding(self, Y,D):
        reg = linear_model.LassoLars(alpha=self.lam)
        reg.fit(D, Y)
        return reg.coef_
    
    def Online_AO(self,Y,batch_size,Tr_Dic,ini_D):
        n,k = ini_D.shape
        D = self.normalize(ini_D)
        A = np.zeros((k, k))
        B = np.zeros((n, k))
        D_r = []
        x_r = []
        conv=[]
        AODLtime = 0
        for yt in self.data_iter(batch_size, Y): 
            AODLtic = time.perf_counter()
            xt = self.Sparse_Coding(yt, D)
            yt=np.squeeze(np.asarray(yt))
            A = A+ np.dot(xt.T,xt)
            B = B+ np.dot(yt, xt)
            D = self.Dic_Update(A, B, D)
            AODLtoc = time.perf_counter()
            error= np.abs(np.sum(np.power(np.matmul(D.T,Tr_Dic),4))/Tr_Dic.shape[1]-1)  
            conv.append(error)
            AODLtime = AODLtime+(AODLtoc-AODLtic)
            D_r.append(D)
            x_r.append(xt)
        self.x = np.array(x_r)
        self.D = np.array(D_r)
        self.conv = np.array(conv)
        self.AODLtime = AODLtime
        #
        return self.x,self.D,self.AODLtime,self.conv
    
    def Batch_AO(self,Y,batch_size,Tr_Dic,ini_D):
        n,k = ini_D.shape
        D = self.normalize(ini_D)
        A = np.zeros((k, k))
        B = np.zeros((n, k))
        for it in range(self.Num_iteration):
            for yt in self.data_iter(batch_size, Y): 
                xt = self.Sparse_Coding(yt, D).T
                A = A+ np.dot(xt,xt.T)
                B = B+ np.dot(yt, xt.T)
                D = self.Dic_Update(A, B, D)
            #error= np.abs(np.sum(np.power(np.matmul(D.T,Tr_Dic),4))/Tr_Dic.shape[1]-1)  
            #conv.append(error)
            self.D = D
        #
        return self.D
        

   
    """
    ## test
theta = 0.3
n = 10# dic_size_row
m=n #dic_size_col
k = 1# num_of_atom for one recovery
lp = 3
r = 10000
noise_var = 0
Num_test = 1
#rconv_sum = np.zeros(Num_iter)
# Start 

for it in range(Num_test):
    
    D0 = np.random.randn(n,n)
    tr_D1, R1 = np.linalg.qr(D0)
    tr_D= tr_D1[:,:m]
    testI = np.matmul(np.transpose(tr_D),tr_D)
    
    B0 = np.random.rand(m,r)
    B = np.copy(B0)
    B[B>theta] = 0
    B[B>0] = 1
    G = np.random.randn(m,r)
    
    X = B * G
    Y = np.matmul(tr_D, X) + np.sqrt(noise_var)*np.random.rand(m,r)
    
    Aini = np.random.rand(n,n)
    A, _ = np.linalg.qr(Aini)
    
    AOonline = Online_AODL(mini_batchsize = 1, lam=0.1)
    rAAO, rconvAO = AOonline.Online_AO(Y,tr_D,A)
 """   