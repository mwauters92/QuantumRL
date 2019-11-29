import numpy as np
import math
import sys
from  scipy.linalg import eigh
import matplotlib.pyplot as plt
from gym import spaces
import tensorflow as tf
import quantum_env as qenv
from spinup.utils.test_policy import load_policy, run_policy


def rew2en(reward, rtype, N=2):
    if rtype == 'energy':
        return -reward
    elif rtype == 'expE':
        return -0.25*np.log(reward)*(N-1)
    elif rtype == 'logE':
        return 1.0-np.exp(-reward)
    else:
        return 0

actType= 'cont'   #bin or cont
model='TFIM'
tau=2*np.pi
P=np.arange(2,20,2)
rew = 'energy'   # energy, logE, expE
Na=20
N=32
ps=2
epochs=512
nstep=8192

dirO = "../Output/"+model+"/ep"+str(epochs)+"_sep"+str(nstep)+"/"
for Nt in P:
    tf.reset_default_graph()
    dt=tau/Nt
    if model == 'SingleSpin':
        env = qenv.SingleSpin(Nt,rew,dt,actType)
        dirOut=dirO+model+actType+"P"+str(Nt)+'_rw'+rtype
    elif model == 'pSpin':
        env = qenv.pSpin(N,ps,Nt,rew,dt,actType)
        dirOut=dirO+'pspin'+"P"+str(Nt)+'_N'+str(Ns)+'_rw'+rtype
    elif model == 'TFIM':
        env_fn = lambda : qenv.TFIM(Ns,Nt,rtype,dt,actType)
        dirOut=dirO+'TFIM'+"P"+str(Nt)+'_N'+str(Ns)+'_rw'+rtype
    else:
        raise ValueError(f'Model not implemented:{model}')
        

    _, get_action = load_policy('./'+dirOut)

    if actType=='bin':
        head='# 1-episode,  2-action, 3-reward, 4-energy'
        data=np.zeros([Na*Nt,4])
        summary=np.zeros([Na+1,3])
        for ep in range(Na):
            o = env.reset()
            for i in range(Nt):
                a = get_action(o)
                o, r, d, _ = env.step(a)
                data[ep*Nt+i,:]=np.array([ep, a, r, rew2en(r,rew,N)])
            summary[ep,:]=np.array([ep,r,rew2en(r,rew,N)])
        summary[-1,:]=np.mean(summary[:-1,:],axis=0)

        np.savetxt(dirOut+"/actions.dat",data, header=head,fmt='%03d  %1d  %.6e  %.6e') 
        head='1-episode,  2-reward 3-energy'
        np.savetxt(dirOut+"/summary.dat",summary, header=head,fmt='%03d  %.6e  %.6e') 

    elif actType=='cont':
        head='# 1-episode,  2-action-gamma, 3-action-beta, 4-reward, 5-energy'
        data=np.zeros([Na*Nt,5])
        summary=np.zeros([Na+1,4])
        for ep in range(Na):
            o = env.reset()
            for i in range(Nt):
                a = np.clip(get_action(o),0,2*np.pi)
                o, r, d, _ = env.step(a)
                data[ep*Nt+i,:]=np.array([ep, a[0],a[1], r, rew2en(r,rew,N)])
            summary[ep,:]=np.array([ep,r,rew2en(r,rew,N),np.sum(data[ep*Nt:(ep+1)*Nt,1:2])])
        summary[-1,:]=summary[:-1,:].mean(axis=0)
        np.savetxt(dirOut+"/actions.dat",data, header=head,fmt='%03d  %.6e  %.6e  %.6e  %.6e') 
        head='# 1-episode,  2-reward 3-energy, 4-time'
        np.savetxt(dirOut+"/summary.dat",summary, header=head,fmt='%03d  %.6e  %.6e  %.6e'  ) 

