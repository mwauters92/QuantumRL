import numpy as np
import math
import sys
from  scipy.linalg import eigh
import matplotlib.pyplot as plt
from gym import spaces
import tensorflow as tf
import quantum_env as qe
from spinup.utils.test_policy import load_policy, run_policy


def rew2en(reward, rtype):
    if rtype == 'energy':
        return -reward
    elif rtype == 'expE':
        return -0.5*np.log(reward)
    elif rtype == 'logE':
        return 1.0-np.exp(-reward)
    else:
        return 0

actType= 'pspin'   #bin or cont
tau=2*np.pi
P=np.arange(2,20,2)
P=[1]
rew = 'energy'   # energy, logE, expE
Na=20
if actType =="bin":
    for Nt in P:
        tf.reset_default_graph()
        dt=tau/Nt
        env = qe.Spin_bin(Nt,dt,rtype=rew,noise=0.2)
        dirOut=actType+"P"+str(Nt)+"_tau2pi_rw"+rew+"_noise0.2"

        _, get_action = load_policy('./'+dirOut)
        head='# 1-episode,  2-action, 3-reward, 4-energy'
        data=np.zeros([Na*Nt,4])
        summary=np.zeros([Na+1,3])
        for ep in range(Na):
            o = env.reset()
            for i in range(Nt):
                a = get_action(o)
                o, r, d, _ = env.step(a)
                data[ep*Nt+i,:]=np.array([ep, a, r, rew2en(r,rew)])
            summary[ep,:]=np.array([ep,r,rew2en(r,rew)])
        summary[-1,:]=np.mean(summary[:-1,:],axis=0)
        np.savetxt(dirOut+"/actions.dat",data, header=head,fmt='%03d  %1d  %.6e  %.6e') 
        head='1-episode,  2-reward 3-energy'
        np.savetxt(dirOut+"/summary.dat",summary, header=head,fmt='%03d  %.6e  %.6e') 

#actType="cont"
if actType =="cont":
    for Nt in P:
        tf.reset_default_graph()
        env = qe.Spin_cont(Nt,rtype=rew,noise=1.0)
        dirOut=actType+"P"+str(Nt)+"_rw"+rew+"_noise1.0"
        #dirOut=actType+"P"+str(Nt)+"_rw"+rew

        _, get_action = load_policy('./'+dirOut)
        head='# 1-episode,  2-action, 3-reward, 4-energy'
        data=np.zeros([Na*Nt,5])
        summary=np.zeros([Na+1,4])
        for ep in range(Na):
            o = env.reset()
            for i in range(Nt):
                a = np.clip(get_action(o),0,2*np.pi)
                o, r, d, _ = env.step(a)
                data[ep*Nt+i,:]=np.array([ep, a[0],a[1], r, rew2en(r,rew)])
            summary[ep,:]=np.array([ep,r,rew2en(r,rew),np.sum(data[ep*Nt:(ep+1)*Nt,1:2])])
        summary[-1,:]=summary[:-1,:].mean(axis=0)
        np.savetxt(dirOut+"/actions.dat",data, header=head,fmt='%03d  %.6e  %.6e  %.6e  %.6e') 
        head='# 1-episode,  2-reward 3-energy, 4-time'
        np.savetxt(dirOut+"/summary.dat",summary, header=head,fmt='%03d  %.6e  %.6e  %.6e'  ) 

if actType =="pspin":
    for Nt in P:
        tf.reset_default_graph()
        env = qe.Pspin_cont(21,2,Nt,rtype=rew,noise=0.0)
        #dirOut=actType+"P"+str(Nt)+"_rw"+rew+"_noise1.0"
        dirOut=actType+"P"+str(Nt)+"_rw"+rew

        _, get_action = load_policy('./'+dirOut)
        head='# 1-episode,  2-action, 3-reward, 4-energy'
        data=np.zeros([Na*Nt,5])
        summary=np.zeros([Na+1,4])
        for ep in range(Na):
            o = env.reset()
            for i in range(Nt):
                a = np.clip(get_action(o),0,2*np.pi)
                o, r, d, _ = env.step(a)
                data[ep*Nt+i,:]=np.array([ep, a[0],a[1], r, rew2en(r,rew)])
            summary[ep,:]=np.array([ep,r,rew2en(r,rew),np.sum(data[ep*Nt:(ep+1)*Nt,1:2])])
        summary[-1,:]=summary[:-1,:].mean(axis=0)
        np.savetxt(dirOut+"/actions.dat",data, header=head,fmt='%03d  %.6e  %.6e  %.6e  %.6e') 
        head='# 1-episode,  2-reward 3-energy, 4-time'
        np.savetxt(dirOut+"/summary.dat",summary, header=head,fmt='%03d  %.6e  %.6e  %.6e'  ) 

