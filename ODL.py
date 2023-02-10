# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 10:07:25 2021

@author: yokoxue
"""

import numpy as np
import time

class ODL_batch:
    def __init__(self,  iterations=100, lp=3):
        self.iterations, self.lp = iterations, lp
        
    def gradient(self, AY, Y, lp):
            if lp % 2 ==1:
                gA = np.matmul(np.multiply(np.power(np.abs(AY),lp-1),np.sign(AY)),np.transpose(Y))/self.sam_size
            else:
                gA = np.matmul(np.power(AY,lp-1),np.transpose(Y))/self.sam_size
            return gA
        
    def proj(self, gA):
            u, s, vh = np.linalg.svd(gA, full_matrices=False)
            An = np.matmul(u,vh)
            return An
        
    
        
    def GPM(self, Y,Tr_Dic,A):
        self.dic_size = Y.shape[0]
        self.num_atom = A.shape[0]
        self.sam_size = Y.shape[1]    
       
        conv = np.zeros(self.iterations)
        for i in range(self.iterations):
           
            conv[i]= np.abs(np.sum(np.power(np.matmul(A,Tr_Dic),4))/self.num_atom-1) 
            AY = np.matmul(A,Y)
            grad = self.gradient(AY, Y, self.lp)
            A = self.proj(grad)
             
            
        self.A = A
        self.conv = conv
        return self.A , self.conv
    
    
class ODL_online:
    def __init__(self,  num_epochs=1, lp=3,theta=0.3 ):
        self.num_epochs, self.lp , self.theta = num_epochs, lp,theta
    
    def data_iter(self, batch_size, Y):
        indices = list(range(self.num_examples))
    # The examples are read at random, in no particular order
       # random.shuffle(indices)
        for i in range(0, self.num_examples, batch_size):
            batch_indices = np.array(indices[i: min(i + batch_size, self.num_examples)])
            yield Y[:,batch_indices]
        
    def gradient(self, AY, Y, lp):
        batch_size = Y.shape[1]
        if lp % 2 ==1:
            gA = np.matmul(np.multiply(np.power(np.abs(AY),lp-1),np.sign(AY)),np.transpose(Y))
        else:
            gA = np.matmul(np.power(AY,lp-1),np.transpose(Y))
        return 1/batch_size*gA
        
    def proj(self, gA):
       # nA = np.linalg.norm(gA,2)
      #  An = gA/nA
        u, _, vh = np.linalg.svd(gA, full_matrices=False)
        An = np.matmul(u,vh)
        return An
    
    def proj_Ball(self, gA):
            u, s, vh = np.linalg.svd(gA, full_matrices=False)
            st = s;
            indlarg = np.where(np.abs(st)>1)
            st[indlarg] =1    
            vht = np.matmul(np.diag(st),vh)
            An = np.matmul(u,vht)
            return An
    """ 
    def FWGap(self, A,Tr_Dic, theta):
       
        # cal mean Gradient
        m = A.shape[0]
        r = 100000
        B0 = np.random.rand(m,r)
        B = copy.copy(B0)
        B[B>theta] = 0
        B[B>0] = 1
        G = np.random.randn(m,r)
        X = B * G
        Y = np.matmul(tr_D, X)
        AY = np.matmul(A,Y)
        Grad_t= self.gradient(AY, Y, self.lp)
        S_t = self.proj(Grad_t)
        D_t = S_t - A
        return np.trace(np.matmul(np.transpose(D_t),Grad_t))/m

    def SGD(self, Y, batch_size, Tr_Dic, A):
        self.dic_size = Y.shape[0]
        self.num_atom = A.shape[0]
        self.num_examples = Y.shape[1]
        #eta = eta_in # learning rate
        conv = []
        totit =0
        Grad = 0
        for i in range(self.num_epochs): 
            it = 1
            for Ys in self.data_iter(batch_size, Y):
                totit = it + i*(self.num_examples/Ys.shape[1])
                rhot = 4/((totit+1)**(1/2))
                gat =  2/((totit+1)**(3/4))
                AY = np.matmul(A,Ys)
                Grad = (1-rhot)*Grad + rhot*self.gradient(AY, Ys, self.lp)
                uA = A + gat * Grad
                A = self.proj_Ball(uA)
                error= np.abs(np.sum(np.power(np.matmul(A,Tr_Dic),4))/self.num_atom-1)  
                conv.append(error)
                it +=1
        self.A = A
        self.conv = np.array(conv)
        return self.A , self.conv
    """
    def NoncvxSFW(self, Y, batch_size, Tr_Dic, A):
        self.dic_size = Y.shape[0]
        self.num_atom = A.shape[0]
        self.num_examples = Y.shape[1]
      #  eta = eta_in# learning rate
        conv = []
        Ar = []
        Grad = 0    
        it = 0
        ODLtime = 0
        for Ys in self.data_iter(batch_size, Y):  
            rhot = 4/((it+1)**(1/2))
            gat = 2/((it+2)**(3/4))
            ODLtic = time.perf_counter()
            AY = np.matmul(A,Ys)
            Grad = (1-rhot)*Grad+rhot*self.gradient(AY, Ys, self.lp)
            A = (1-gat)*A + gat*self.proj(Grad)
            A = self.proj(A)
            error= np.abs(np.sum(np.power(np.matmul(A,Tr_Dic),4))/self.num_atom-1)  
            ODLtoc = time.perf_counter()
            #  error= self.FWGap(A,Tr_Dic, self.theta)
            conv.append(error)
            Ar.append(A)
            ODLtime = ODLtime+(ODLtoc-ODLtic)
            it +=1
        self.A = np.array(Ar)
        self.conv = np.array(conv)
        self.ODLtime = ODLtime
        return self.A , self.ODLtime, self.conv
    
    def SFW(self, Y, batch_size, Tr_Dic, A):
        self.dic_size = Y.shape[0]
        self.num_atom = A.shape[0]
        self.num_examples = Y.shape[1]
      #  eta = eta_in# learning rate
        conv = []
        Ar = []
        Grad = 0    
        it = 0
        ODLtime = 0
        for Ys in self.data_iter(batch_size, Y): 
            rhot = 4/((it+8)**(2/3))
            gat = 2/((it+8)**(1))
            ODLtic = time.perf_counter()
            AY = np.matmul(A,Ys)
            Grad = (1-rhot)*Grad+rhot*self.gradient(AY, Ys, self.lp)
            A = (1-gat)*A + gat*self.proj(Grad)
         #   A = self.proj(A)
            error= np.abs(np.sum(np.power(np.matmul(self.proj(A),Tr_Dic),4))/self.num_atom-1)
            ODLtoc = time.perf_counter()
            #  error= self.FWGap(A,Tr_Dic, self.theta)
            conv.append(error)
            Ar.append(A)
            ODLtime = ODLtime+(ODLtoc-ODLtic)
            it +=1
        self.A = np.array(Ar)
        self.conv = np.array(conv)
        self.ODLtime = ODLtime
        return self.A , self.ODLtime, self.conv
    