# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 15:18:06 2021

@author: XUEYe
"""

# -*- coding: utf-8 -*-




import numpy as np
import copy
from ODL import ODL_online
from oAODL import Online_AODL
import time


""" parameters """
theta = 0.3
n = 10 # dic_size_row
m=n #dic_size_col
k = 1# num_of_atom for one recovery
lp = 3
r = 30000
noise_var = 0
Num_test = 100
Num_iter = r
split = 30000

batch_size_fw0=1

rconvl3_Nsfw_sum0 =  np.zeros(int(Num_iter/batch_size_fw0))
rconvl3_sfwcvx_sum0 =  np.zeros(int(Num_iter/batch_size_fw0))
rconvl4_Nsfw_sum0 =  np.zeros(int(Num_iter/batch_size_fw0))
rconvAO_sum0 =  np.zeros(int(Num_iter/batch_size_fw0))
batch_size_fw1=10

rconvl3_Nsfw_sum1 =  np.zeros(int(Num_iter/batch_size_fw1))
rconvl3_sfwcvx_sum1 =  np.zeros(int(Num_iter/batch_size_fw1))
rconvl4_Nsfw_sum1 =  np.zeros(int(Num_iter/batch_size_fw1))
rconvAO_sum1 =  np.zeros(int(Num_iter/batch_size_fw1))
""" Start """

for it in range(Num_test):
    tic = time.time()  
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
  
    
    oAODL = Online_AODL(lam = 0.1,Num_iteration = 1)
    xAO_0,DAO_0,rconvAODL0,_ = oAODL.Online_AO(Y,batch_size_fw0,tr_D,np.transpose(A[:,:m]))
    xAO_1,DAO_1,rconvAODL1,_ = oAODL.Online_AO(Y,batch_size_fw1,tr_D,np.transpose(A[:,:m]))
    
    oODLl3 = ODL_online(num_epochs = 1, lp = 3, theta = theta)
    #rAsgdl3, rconvsgdl3 = Sphl3.SGD(Y,batch_size_fw,tr_D,np.transpose(A[:,:m]))
    _,_, rconvNsfwl30 = oODLl3.NoncvxSFW(Y,batch_size_fw0,tr_D,np.transpose(A[:,:m]))
    _,_, rconvNsfwl31 = oODLl3.NoncvxSFW(Y,batch_size_fw1,tr_D,np.transpose(A[:,:m]))
   
    #rAspcal3, rconvspcal3 = Sphl3.SPCA(Y,batch_size,tr_D,np.transpose(A[:,:m]),0.001)
    _,_, rconvsfwcvxl30 = oODLl3.SFW(Y,batch_size_fw0,tr_D,np.transpose(A[:,:m]))
    _,_, rconvsfwcvxl31 = oODLl3.SFW(Y,batch_size_fw1,tr_D,np.transpose(A[:,:m]))
    
    oODLl4 = ODL_online(num_epochs = 1, lp = 4, theta = theta)
    _,_, rconvNsfwl40 = oODLl4.NoncvxSFW(Y,batch_size_fw0,tr_D,np.transpose(A[:,:m]))
    _,_, rconvNsfwl41 = oODLl4.NoncvxSFW(Y,batch_size_fw1,tr_D,np.transpose(A[:,:m]))
    
    
    
    #rconvl3_sgd_sum = rconvl3_sgd_sum + rconvsgdl3/Num_test
    rconvl3_Nsfw_sum0 = rconvl3_Nsfw_sum0 + rconvNsfwl30/Num_test
    rconvl3_sfwcvx_sum0 = rconvl3_sfwcvx_sum0 + rconvsfwcvxl30/Num_test
    rconvl4_Nsfw_sum0 = rconvl4_Nsfw_sum0 + rconvNsfwl40/Num_test
    rconvAO_sum0 = rconvAO_sum0+rconvAODL0 /Num_test
  
    rconvl3_Nsfw_sum1 = rconvl3_Nsfw_sum1 + rconvNsfwl31/Num_test
    rconvl3_sfwcvx_sum1 = rconvl3_sfwcvx_sum1 + rconvsfwcvxl31/Num_test
    rconvl4_Nsfw_sum1 = rconvl4_Nsfw_sum1 + rconvNsfwl41/Num_test
    rconvAO_sum1 = rconvAO_sum1+rconvAODL1 /Num_test
    #rconvl3_spca_sum = rconvl3_spca_sum + rconvspcal3/Num_test
    toc = time.time()
    print(" test:{},time:{:.2f}\n,conv1:{:.2e},conv2:{:.2e},conv3:{:.2e},conv4:{:.2e}".format(it , toc-tic, rconvNsfwl31[-1],rconvsfwcvxl31[-1],rconvNsfwl41[-1],rconvAODL1[-1]))
   
Conv_diff = np.stack((rconvl3_Nsfw_sum0[:3000], rconvl3_sfwcvx_sum0[:3000],\
                         rconvl4_Nsfw_sum0[:3000], rconvAO_sum0 [:3000],\
                             rconvl3_Nsfw_sum1[:3000], rconvl3_sfwcvx_sum1[:3000],\
                              rconvl4_Nsfw_sum1[:3000], rconvAO_sum1 [:3000]))
np.savetxt('Conv_diffbatch.csv', Conv_diff, delimiter=',')
"""
##plot emperical convergence
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator
x = np.arange(Num_iter)
fig, ax = plt.subplots()
#x1 = np.arange(len(rconvl3_sgd_sum))
x2 = np.arange(len(rconvl3_Nsfw_sum0))
x3 = np.arange(len(rconvl3_sfwcvx_sum0))
#ax.semilogy(np.arange(len(rconvl3_sum)),rconvl3_sum,'-p',label='$\ell_3$GPM(fullbatch)',linewidth=2.0)
#ax.semilogy(x1[:3000], rconvl3_sgd_sum[:3000], '-',label='ProxSGD',linewidth=2.0)
ax.semilogy(x2[:3000], rconvl3_Nsfw_sum0[:3000], '--',linewidth=2.0,color='#054E9F')  
ax.semilogy(x3[:3000], rconvl4_Nsfw_sum0[:3000], '--',linewidth=2.0,color = 'coral' )  
x4 = np.arange(len(rconvl3_Nsfw_sum1))
x5 = np.arange(len(rconvl3_sfwcvx_sum1))
#ax.semilogy(np.arange(len(rconvl3_sum)),rconvl3_sum,'-p',label='$\ell_3$GPM(fullbatch)',linewidth=2.0)
#ax.semilogy(x1[:3000], rconvl3_sgd_sum[:3000], '-',label='ProxSGD',linewidth=2.0)
ax.semilogy(x4[:3000], rconvl3_Nsfw_sum1[:3000], '-.',label='NoncvxSFW',linewidth=2.0,color='#054E9F')  
ax.semilogy(x5[:3000], rconvl4_Nsfw_sum1[:3000], '-.',label='SFW',linewidth=2.0,color ='coral')  
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
"""