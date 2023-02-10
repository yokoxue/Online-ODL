# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 17:34:20 2021

@author: XUEYe
"""




import numpy as np
import copy

class GPM_lp_Maximization_sphere:
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
    
    
class sto_lp_Maximization_O:
    def __init__(self,  num_epochs=1, lp=3):
        self.num_epochs, self.lp = num_epochs, lp
    
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
    
    def SFW(self, Y, batch_size, Tr_Dic, A):
        self.dic_size = Y.shape[0]
        self.num_atom = A.shape[0]
        self.num_examples = Y.shape[1]
      #  eta = eta_in# learning rate
        conv = []
        totit = 0
        Grad = 0
        for i in range(self.num_epochs):
            
            it = 1
            for Ys in self.data_iter(batch_size, Y): 
                totit = it + i*(self.num_examples/Ys.shape[1])
                rhot = 4/((totit+1)**(1/2))
                gat = 2/((totit+2)**(3/4))
                AY = np.matmul(A,Ys)
                Grad = (1-rhot)*Grad+rhot*self.gradient(AY, Ys, self.lp)
                A = (1-gat)*A + gat*self.proj(Grad)
                A = self.proj(A)
                error= np.abs(np.sum(np.power(np.matmul(A,Tr_Dic),4))/self.num_atom-1)  
                conv.append(error)
                it +=1
        self.A = A
        self.conv = np.array(conv)
        return self.A , self.conv
    
    def SFW_cvx(self, Y, batch_size, Tr_Dic, A):
        self.dic_size = Y.shape[0]
        self.num_atom = A.shape[0]
        self.num_examples = Y.shape[1]
      #  eta = eta_in# learning rate
        conv = []
        totit = 0
        Grad = 0
        for i in range(self.num_epochs):
            
            it = 1
            for Ys in self.data_iter(batch_size, Y): 
                totit = it + i*(self.num_examples/Ys.shape[1])
                rhot = 4/((totit+8)**(2/3))
                gat = 2/((totit+8)**(1))
                AY = np.matmul(A,Ys)
                Grad = (1-rhot)*Grad+rhot*self.gradient(AY, Ys, self.lp)
                A = (1-gat)*A + gat*self.proj(Grad)
                A = self.proj(A)
                error= np.abs(np.sum(np.power(np.matmul(A,Tr_Dic),4))/self.num_atom-1)  
                conv.append(error)
                it +=1
        self.A = A
        self.conv = np.array(conv)
        return self.A , self.conv
    
    
    
    

""" parameters """
theta = 0.3
n = 50 # dic_size_row
m=n #dic_size_col
k = 1# num_of_atom for one recovery
lp = 3
r = 100000
noise_var = 0
Num_test = 100
Num_iter = r
split = 100000
batch_size = int(r/split)

batch_size_fw0=10
#rconvl3_sgd_sum =  np.zeros(int(Num_iter/batch_size_fw))
rconvl3_sfw_sum0 =  np.zeros(int(Num_iter/batch_size_fw0))
rconvl3_sfwcvx_sum0 =  np.zeros(int(Num_iter/batch_size_fw0))
batch_size_fw1=10
#rconvl3_sgd_sum =  np.zeros(int(Num_iter/batch_size_fw))
rconvl3_sfw_sum1 =  np.zeros(int(Num_iter/batch_size_fw1))
rconvl3_sfwcvx_sum1 =  np.zeros(int(Num_iter/batch_size_fw1))

""" Start """

for it in range(Num_test):
    
    D0 = np.random.randn(n,n)
    tr_D1, R1 = np.linalg.qr(D0)
    tr_D= tr_D1[:,:m]
    testI = np.matmul(np.transpose(tr_D),tr_D)
    
    B0 = np.random.rand(m,r)
    B = copy.copy(B0)
    B[B>theta] = 0
    B[B>0] = 1
    G = np.random.randn(m,r)
    
    X = B * G
    Y = np.matmul(tr_D, X) + np.sqrt(noise_var)*np.random.rand(m,r)
    
    Aini = np.random.rand(n,n)
    A, _ = np.linalg.qr(Aini)
    '''
    gpml3 = GPM_lp_Maximization_sphere(iterations = Num_iter, lp = 3)
    rAl3, rconvl3 = gpml3.GPM(Y,tr_D,np.transpose(A[:,:m]))
    
    gpml4 = GPM_lp_Maximization_sphere(iterations = Num_iter, lp = 4)
    rAl4, rconvl4 = gpml4.GPM(Y,tr_D, np.transpose(A[:,:m]))
   
    rconvl3_sum = rconvl3_sum + rconvl3/Num_test
    #rconvl4_sum = rconvl4_sum + rconvl4/Num_test
     '''
    Sphl3 = sto_lp_Maximization_O(num_epochs = 1, lp = 3)
    #rAsgdl3, rconvsgdl3 = Sphl3.SGD(Y,batch_size_fw,tr_D,np.transpose(A[:,:m]))
    rAsfwl30, rconvsfwl30 = Sphl3.SFW(Y,batch_size_fw0,tr_D,np.transpose(A[:,:m]))
    rAsfwcvxl30, rconvsfwcvxl30 = Sphl3.SFW_cvx(Y,batch_size_fw0,tr_D,np.transpose(A[:,:m]))
    #rAspcal3, rconvspcal3 = Sphl3.SPCA(Y,batch_size,tr_D,np.transpose(A[:,:m]),0.001)
   # rAsfwl31, rconvsfwl31 = Sphl3.SFW(Y,batch_size_fw1,tr_D,np.transpose(A[:,:m]))
    #rAsfwcvxl31, rconvsfwcvxl31 = Sphl3.SFW_cvx(Y,batch_size_fw1,tr_D,np.transpose(A[:,:m]))
   
    '''
   Sphl4 = sto_lp_Maximization_sphere(num_epochs = Num_iter, lp = 4)
    rAsgdl4, rconvsgdl4 = Sphl4.SGD(Y,batch_size, tr_D, np.transpose(A[:,:m]), 0.2)
    rAsfwl4, rconvsfwl4 = Sphl4.SFW(Y,batch_size, tr_D, np.transpose(A[:,:m]), 0.2)
    rAspcal4, rconvspcal4 = Sphl4.SPCA(Y,batch_size, tr_D, np.transpose(A[:,:m]),0.001)
   ''' 
    
    #rconvl3_sgd_sum = rconvl3_sgd_sum + rconvsgdl3/Num_test
    rconvl3_sfw_sum0 = rconvl3_sfw_sum0 + rconvsfwl30/Num_test
    rconvl3_sfwcvx_sum0 = rconvl3_sfwcvx_sum0 + rconvsfwcvxl30/Num_test
   # rconvl3_sfw_sum1 = rconvl3_sfw_sum1 + rconvsfwl31/Num_test
    #rconvl3_sfwcvx_sum1 = rconvl3_sfwcvx_sum1 + rconvsfwcvxl31/Num_test
    #rconvl3_spca_sum = rconvl3_spca_sum + rconvspcal3/Num_test
    '''
    rconvl4_sgd_sum = rconvl4_sgd_sum + rconvsgdl4/Num_test
    rconvl4_sfw_sum = rconvl4_sfw_sum + rconvsfwl4/Num_test
    rconvl4_spca_sum = rconvl4_spca_sum + rconvspcal4/Num_test
   '''
    l3_conv_diff = np.stack((rconvl3_sfw_sum0[:3000], rconvl3_sfwcvx_sum0[:3000]))
    np.savetxt('l3_conv_diffN.csv', l3_conv_diff, delimiter=',')

"""plot emperical convergence"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator
x = np.arange(Num_iter)
fig, ax = plt.subplots()
#x1 = np.arange(len(rconvl3_sgd_sum))
x2 = np.arange(len(rconvl3_sfw_sum0))
x3 = np.arange(len(rconvl3_sfwcvx_sum0))
#ax.semilogy(np.arange(len(rconvl3_sum)),rconvl3_sum,'-p',label='$\ell_3$GPM(fullbatch)',linewidth=2.0)
#ax.semilogy(x1[:3000], rconvl3_sgd_sum[:3000], '-',label='ProxSGD',linewidth=2.0)
ax.semilogy(x2[:3000], rconvl3_sfw_sum0[:3000], '--',linewidth=2.0,color='#054E9F')  
ax.semilogy(x3[:3000], rconvl3_sfwcvx_sum0[:3000], '--',linewidth=2.0,color = 'coral' )  
#x4 = np.arange(len(rconvl3_sfw_sum1))
#x5 = np.arange(len(rconvl3_sfwcvx_sum1))
#ax.semilogy(np.arange(len(rconvl3_sum)),rconvl3_sum,'-p',label='$\ell_3$GPM(fullbatch)',linewidth=2.0)
#ax.semilogy(x1[:3000], rconvl3_sgd_sum[:3000], '-',label='ProxSGD',linewidth=2.0)
#ax.semilogy(x4[:3000], rconvl3_sfw_sum1[:3000], '-.',label='NoncvxSFW',linewidth=2.0,color='#054E9F')  
#ax.semilogy(x5[:3000], rconvl3_sfwcvx_sum1[:3000], '-.',label='SFW',linewidth=2.0,color ='coral')  
#ax.semilogy(x, rconvl4_sfw_sum , '-o',label='$\ell_4$Online_FW($\eta = 0.2$)',linewidth=2.0)  
#ax.semilogy(x, rconvl3_spca_sum , '-s',label='$\ell_3$Online_PCA($\eta_M = 0.001$)',linewidth=2.0)

rect1 = patches.Rectangle((300,0.002),200,0.018,linewidth=1.5,edgecolor='#054E9F',facecolor='none')
ax.add_artist(rect1)
plt.annotate('NoncvxSFW \n (mini-batch size = 1,10)',fontsize=14, weight= 'bold',family='Times New Roman', xy=(250, 0.0019), xytext=(0,0.03),
               arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3",lw=2,color='#054E9F'))

rect2 = patches.Rectangle((1000,0.04),200,0.1,linewidth=1.5,edgecolor='coral',facecolor='none')
ax.add_artist(rect2)
plt.annotate('SFW \n (mini-batch size = 1,10)',fontsize=14, weight= 'bold',family='Times New Roman', xy=(950, 0.039), xytext=(800,0.2),
               arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3",lw=2,color='coral'))



 
#设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=18)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
 
#设置横纵坐标的名称以及对应字体格式
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 23,
}
plt.xlabel('Number of Iterations',font2)
plt.ylabel('Error',font2)
plt.grid(True,which="both")
#plt.title("Convergence over the orthogonal group ",fontdict={'family' : 'Times New Roman', 'size'   : 30})
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig('oGaveFw1_O.eps',format='eps', bbox_inches = 'tight')