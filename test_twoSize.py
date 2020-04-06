import numpy as np
#import math
import sys
#from  scipy.linalg import eigh
#import matplotlib.pyplot as plt
from gym import spaces
#import tensorflow as tf
from tensorflow import reset_default_graph
import quantum_env as qenv
from spinup.utils.test_policy import load_tf_policy, run_policy, load_policy_and_env
import argparse
from scipy.optimize import minimize
import time

def rew2en(reward, rtype, N=2):
    if rtype == 'energy':
        return -reward
    elif rtype == 'expE':
        return -0.25*np.log(reward)*(N)
    elif rtype == 'logE':
        return 1.0-np.exp(-reward)
    else:
        return 0


parser = argparse.ArgumentParser(description="Validation of Reinforcement Learing policies for QAOA")
parser.add_argument('--model', default="SingleSpin", type=str, help="Model to study with QAOA+RL")
parser.add_argument('--obs', default="tomography", type=str, help="observables seen by the Reinforcement Learing agent")
parser.add_argument('--actType', default="cont", type=str, help="action space of the RL agent: cont or bin")
parser.add_argument('--tau', default=10, type=float, help="total time for binary actions")
parser.add_argument('--Pvars', default=[2,3,1], nargs=3, type=int, help="episodes lengths (start,stop,step)")
parser.add_argument('--network', default=[16,16],nargs='+', type=int, help="Number of neurons in each layer")
parser.add_argument('--deterministic', default = False , type=bool, help="Validation with deterministic actions")
parser.add_argument('--rtype', default="energy", type=str, help="reward function (energy, expE or logE")
parser.add_argument('--epochs', default=512, type=int, help="Number of epochs of the training process")
parser.add_argument('--nstep', default=1024, type=int, help="Number of steps per epoch")
parser.add_argument('--nvalidation', default=20, type=int, help="Number of validation episodes")
parser.add_argument('--N_act', default=32, type=int, help="System size of the trained model")
parser.add_argument('--N_rew', default=128, type=int, help="System size of the test model")
parser.add_argument('--ps', default=2, type=int, help="Interaction rank (only for pSpin)")
parser.add_argument('--hfield', default=0, type=float, help="Transverse field for target state")
parser.add_argument('--plotP', default=False, type=bool, help="if True prints a file with the state value function of in the observation space")
parser.add_argument('--noise', default=0, type=float, help="Noise on the initial state")
parser.add_argument('--seed', default = 812453, type=int, help="Seed for the RandomTFIM model")
parser.add_argument('--local_opt', default = False, type=bool, help="perform local optimization after RL test")

args = parser.parse_args()


actType = args.actType                 # action type: bin, cont
model = args.model                   # model : pSpin, SingleSpin, TFIM
tau = args.tau                          # total time used in the binary action process
P = np.arange(args.Pvars[0],args.Pvars[1],args.Pvars[2])             # list of episode lenghts 
measured_obs = args.obs
rtype = args.rtype                 #reward types: energy, logE, expE
epochs = args.epochs                      # number of epochs
nstep = args.nstep                      # steps per episodes
layers = args.network
deterministic_act = args.deterministic
noise = args.noise
seed = args.seed
local_opt = args.local_opt

if measured_obs == 'Hobs':
    plotSValue = args.plotP
else:
    plotSValue = False
print("value ", plotSValue)
# physical parameters
Ns_act = args.N_act
Ns_rew = args.N_rew
ps = args.ps                      # interaction rank of the pSpin model
hfield = args.hfield
Na = args.nvalidation

def set_couplings( N, seed):
    if seed > 1 :
        couplings = np.random.RandomState(seed=seed).random(N)
    elif seed == 1 :
        couplings = np.ones(N)
    else :
        couplings = np.random.RandomState(seed=None).random(N)
    return couplings

if noise == 0 :
    if hfield > 0:
        dirO = "../Output/"+model+"_g"+str(hfield)+"/ep"+str(epochs)+"_sep"+str(nstep)+"/"
    else:
        dirO = "../Output/"+model+"/ep"+str(epochs)+"_sep"+str(nstep)+"/"
else :
    if hfield > 0:
        dirO = "../Output/"+model+"N_g"+str(hfield)+"/ep"+str(epochs)+"_sep"+str(nstep)+"/"
    else:
        dirO = "../Output/"+model+"N/ep"+str(epochs)+"_sep"+str(nstep)+"/"


for Nt in P:
    reset_default_graph()
    dt=tau/Nt
    if model == 'pSpin':
        env_act = qenv.pSpin(Ns_act,ps,Nt,rtype,dt,actType,measured_obs=measured_obs, g_target=hfield ,noise=noise)
        env_rew = qenv.pSpin(Ns_rew,ps,Nt,rtype,dt,actType,measured_obs=measured_obs, g_target=hfield ,noise=noise)
        dirOut=dirO+'pspin'+"P"+str(Nt)+'_N'+str(Ns_act)+'_rw'+rtype
        gs_energy = -Ns
    elif model == 'TFIM':
        env_act = qenv.TFIM(Ns_act,Nt,rtype,dt,actType,measured_obs=measured_obs, g_target=hfield ,noise=noise)
        env_rew = qenv.TFIM(Ns_rew,Nt,rtype,dt,actType,measured_obs=measured_obs, g_target=hfield ,noise=noise)
        dirOut=dirO+'TFIM'+"P"+str(Nt)+'_N'+str(Ns_act)+'_rw'+rtype
        gs_energy = -Ns_rew
    elif model == 'RandomTFIM':
        J_couplings = set_couplings(Ns_act, seed)
        env_act = qenv.RandomTFIM(Ns_act,J_couplings,Nt,rtype,dt,actType,measured_obs=measured_obs, g_target=hfield ,noise=noise)
        #env_act = qenv.RandomTFIM(Ns_rew,J_couplings,Nt,rtype,dt,actType,measured_obs=measured_obs, g_target=hfield ,noise=noise,seed=1)
        dirOut=dirO+'RandomIsing'+"P"+str(Nt)+'_N'+str(Ns_act)+'_rw'+rtype
        gs_energy = -J_couplings.sum()
    else:
        raise ValueError(f'Model not implemented:{model}')





    dirOut += '/'+measured_obs+'/network'+str(layers[0])+'x'+str(layers[1])
    print(deterministic_act, plotSValue)
    #_, get_action, get_value = load_tf_policy('./'+dirOut,deterministic=deterministic_act, valueFunction=plotSValue)
    _, get_action = load_policy_and_env('./'+dirOut,deterministic=deterministic_act)
    get_value = None

    if actType=='cont':
        head='# 1-episode,  2-action-gamma, 3-action-beta, 4-reward, 5-energy'
        data=np.zeros([Na*Nt,5])
        summary=np.zeros([Na+1,5])
        for ep in range(Na):
            if model == 'RandomTFIM':
                J_couplings = set_couplings(Ns_rew, 0)
                env_rew = qenv.RandomTFIM(Ns_rew,J_couplings,Nt,rtype,dt,actType,measured_obs=measured_obs, g_target=hfield ,noise=noise)
                gs_energy = -J_couplings.sum()
                print(gs_energy)

            if local_opt: f_grad = lambda x : env_rew.get_fullEvo(x, grad=True)
            o_act = env_act.reset()
            o_rew = env_rew.reset()
            x_guess = np.zeros(2*Nt)
            for i in range(Nt):
                a = np.clip(get_action(o_act),0,2*np.pi)
                x_guess[i]=a[0]
                x_guess[i+Nt]=a[1]
                o_rew, r_rew, d_rew, _ = env_rew.step(a)
                o_act, r_act, d_act, _ = env_act.step(a)
                data[ep*Nt+i,:]=np.array([ep, a[0],a[1], r_rew, rew2en(r_rew,rtype,Ns_rew)])
            #summary[ep,:]=np.array([ep,r_rew,rew2en(r_rew,rew,N),np.sum(data[ep*Nt:(ep+1)*Nt,1:2])])
            if local_opt:
                res = minimize(env_rew.get_fullEvo, x_guess, method="BFGS", jac=f_grad, tol=1e-5, options={'disp': True, 'maxiter': 1e5, 'gtol': 1e-5})
                data[ep*Nt:(ep+1)*Nt,1] = res.x[:Nt]
                data[ep*Nt:(ep+1)*Nt,2] = res.x[Nt:]
                summary[ep,:]=np.array([ep,r_rew,(rew2en(r_rew,rtype,Ns_rew)-gs_energy)/(-2*gs_energy),res.x.sum(), (res.fun-gs_energy)/(-2*gs_energy)])
            else:
                summary[ep,:]=np.array([ep,r_rew,(rew2en(r_rew,rtype,Ns_rew)-gs_energy)/(-2*gs_energy),np.sum(data[ep*Nt:(ep+1)*Nt,1:2]), 0 ])

            #summary[ep,:]=np.array([ep,r_rew,(rew2en(r_rew,rtype,Ns_rew)-gs_energy)/(-2*gs_energy),np.sum(data[ep*Nt:(ep+1)*Nt,1:2])])
        #
        summary[-1,:]=summary[:-1,:].mean(axis=0)
        summary[-1,0]=summary[:-1,2].std()
        np.savetxt(dirOut+"/actions_rew.dat",data, header=head,fmt='%03d  %.6e  %.6e  %.6e  %.6e') 
        head='# 1-episode,  2-reward 3-energy, 4-time'
        np.savetxt(dirOut+"/summary_rew.dat",summary, header=head,fmt='%03d  %.6e  %.6e  %.6e %.6e'  ) 

