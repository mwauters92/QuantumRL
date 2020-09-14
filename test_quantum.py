import numpy as np
import math
import sys
from  scipy.linalg import eigh
import matplotlib.pyplot as plt
from gym import spaces
import tensorflow as tf
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
parser.add_argument('--N', default=32, type=int, help="System size (only for pSpin and TFIM)")
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
Ns = args.N
ps = args.ps                      # interaction rank of the pSpin model
hfield = args.hfield
Na = args.nvalidation

'''def set_couplings( N, seed):
    if seed > 1 :
        couplings = np.random.RandomState(seed=seed).random(N)
    elif seed == 1 :
        couplings = np.ones(N)
    else :
        couplings = np.random.RandomState(seed=None).random(N)
    return couplings
'''

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
    tf.reset_default_graph()
    dt=tau/Nt
    if model == 'SingleSpin':
        env = qenv.SingleSpin(Nt,rtype,dt,actType,noise=noise, g_target=hfield)
        dirOut=dirO+model+actType+"P"+str(Nt)+'_rw'+rtype
        gs_energy = -1
    elif model == 'pSpin':
        env = qenv.pSpin(Ns,ps,Nt,rtype,dt,actType,measured_obs=measured_obs, g_target=hfield ,noise=noise)
        dirOut=dirO+'pspin'+"P"+str(Nt)+'_N'+str(Ns)+'_rw'+rtype
        gs_energy = np.linalg.eigvalsh(env.H_target)[0]
        #gs_energy = -Ns
    elif model == 'TFIM':
        env = qenv.TFIM(Ns,Nt,rtype,dt,actType,measured_obs=measured_obs, g_target=hfield ,noise=noise)
        dirOut=dirO+'TFIM'+"P"+str(Nt)+'_N'+str(Ns)+'_rw'+rtype
        gs_energy = -Ns
        Emax = Ns
    elif model == 'RandomTFIM':
        J_couplings = qenv.set_couplings(Ns, seed)
        env = qenv.RandomTFIM(Ns,J_couplings,Nt,rtype,dt,actType,measured_obs=measured_obs, g_target=hfield ,noise=noise,seed=1)
        dirOut=dirO+'RandomIsing'+"P"+str(Nt)+'_N'+str(Ns)+'_rw'+rtype
        gs_energy = -J_couplings.sum()
        Emax = - gs_energy
    elif model == 'SKglass':
        J_couplings = qenv.set_couplings(Ns, seed, model = 'SKglass')
        env = qenv.SKglass(Ns,J_couplings,Nt,rtype,dt,actType,measured_obs=measured_obs, g_target=hfield ,noise=noise,seed=1)
        dirOut=dirO+'SKglass'+"P"+str(Nt)+'_N'+str(Ns)+'_rw'+rtype
        gs_energy = env.Hz.min()
        Emax = env.Hz.max()
        print(gs_energy)
    else:
        raise ValueError(f'Model not implemented:{model}')
    
    if local_opt: f_grad = lambda x : env.get_fullEvo(x, grad=True)

    #dirOut += '/network'+str(layers[0])+'x'+str(layers[1])        
    dirOut += '/'+measured_obs+'/network'+str(layers[0])+'x'+str(layers[1])
    print(deterministic_act, plotSValue) 
    #_, get_action, get_value = load_tf_policy('./'+dirOut,deterministic=deterministic_act, valueFunction=plotSValue)
    _, get_action = load_policy_and_env('./'+dirOut,deterministic=deterministic_act)
    get_value = None
    if actType=='bin':
        head='# 1-episode,  2-action, 3-reward, 4-energy'
        data=np.zeros([Na*Nt,4])
        summary=np.zeros([Na+1,3])
        for ep in range(Na):
            o = env.reset()
            for i in range(Nt):
                a = np.clip(get_action(o), 0, np.pi*2)
                o, r, d, _ = env.step(a)
                data[ep*Nt+i,:]=np.array([ep, a, r, rew2en(r,rtype,Ns)])
            summary[ep,:]=np.array([ep,r,rew2en(r,rtype,Ns)])
        summary[-1,:]=np.mean(summary[:-1,:],axis=0)

        np.savetxt(dirOut+"/actions.dat",data, header=head,fmt='%03d  %1d  %.6e  %.6e') 
        head='1-episode,  2-reward 3-energy'
        np.savetxt(dirOut+"/summary.dat",summary, header=head,fmt='%03d  %.6e  %.6e') 

    elif actType=='cont':
        data=np.zeros([Na*Nt,5])
        data_opt=np.zeros([Na*Nt,2])
        dynamics=np.zeros([Na*Nt,1+env.obs_shape[0]])
        summary=np.zeros([Na+1,5])
        result_locOpt=[]

        #t_QA=0 #DBG
        #t_RL=0 #DBG
        for ep in range(Na):
            """if model == 'RandomTFIM' and ep > 0:
                J_couplings = set_couplings(Ns, 0)
                env = qenv.RandomTFIM(Ns,J_couplings,Nt,rtype,dt,actType,measured_obs=measured_obs, g_target=hfield ,noise=noise,seed=1)
                gs_energy = -J_couplings.sum()"""
            o = env.reset()
            x_guess = np.zeros(2*Nt)
            x_o = np.zeros([Nt,2])
            for i in range(Nt):
                a = np.clip(get_action(o),0,2*np.pi)
                x_guess[i]=a[0]
                x_guess[i+Nt]=a[1]
                o, r, d, _ = env.step(a)
                #t_QA += t0 - time.time()
                data[ep*Nt+i,:]=np.array([ep, a[0],a[1], r, rew2en(r,rtype,Ns)])
                dynamics[ep*Nt+i,:]=np.concatenate(([ep],o))
            
            if local_opt:
                res = minimize(env.get_fullEvo, x_guess, method="BFGS", jac=f_grad, tol=1e-5, options={'disp': True, 'maxiter': 1e5, 'gtol': 1e-5})
                data_opt[ep*Nt:(ep+1)*Nt,0] = res.x[:Nt]
                data_opt[ep*Nt:(ep+1)*Nt,1] = res.x[Nt:]
                summary[ep,:]=np.array([ep,rew2en(r,rtype,env.N)/Ns,(rew2en(r,rtype,env.N)-gs_energy)/(Emax - gs_energy),np.sum(data[ep*Nt:(ep+1)*Nt,1:2]), (res.fun-gs_energy)/(Emax-gs_energy)])
                result_locOpt.append([res.fun, res.x.sum(), res.nit, res.nfev])
            else:
                summary[ep,:]=np.array([ep,rew2en(r,rtype,env.N)/Ns,(rew2en(r,rtype,env.N)-gs_energy)/(Emax - gs_energy),np.sum(data[ep*Nt:(ep+1)*Nt,1:2]), 0 ])
        
        if local_opt: result_locOpt.append(np.array(result_locOpt).mean(axis=0))
        summary[-1,:]=summary[:-1,:].mean(axis=0)
        summary[-1,0]=summary[:-1,2].min()
        summary[-1,3]=summary[:-1,4].min()

        head='# 1-episode,  2-action-gamma, 3-action-beta, 4-reward, 5-energy'
        np.savetxt(dirOut+"/actions.dat",data, header=head,fmt='%03d  %.6e  %.6e  %.6e  %.6e') 
        
        if local_opt:
            head='# 1-action-gamma, 2-action-beta'
            np.savetxt(dirOut+"/actions_opt.dat",data_opt, header=head,fmt='%.6e  %.6e') 

            head='# 1-min_fun, 2-evo time, 3-Niter, 4-fun eval'
            np.savetxt(dirOut+"/local_opt.dat",result_locOpt, header=head,fmt='%.6e  %.6e  %.6e  %.6e') 

        head='# 1-action-gamma, 2-action-beta'
        np.savetxt(dirOut+"/actions_opt.dat",data_opt, header=head,fmt='%.6e  %.6e') 

        head='# 1-episode, 2-reward, 3-res energy, 4-evo time, 5-loc opt'
        np.savetxt(dirOut+"/summary.dat",summary, header=head,fmt='%.6e  %.6e  %.6e  %.6e  %.6e'  ) 

        head='# 1-episode,  2-s_x, 3-szsz, 4- corr 2'
        np.savetxt(dirOut+"/dynamics.dat",dynamics, header=head  ) 

        if plotSValue:
            value = []
            for ozz in np.linspace(-1,1,100):
                for ox in np. linspace(0,1,100):
                    o=np.array([ox,ozz])
                    #a = np.clip(get_action(o), 0, np.pi*2)
                    a = get_action(o)
                    value.append(np.array([ox,ozz,a[0], a[1]]))
            np.savetxt(dirOut+"/valueFunction.dat",value, header=head,fmt='%.6e  %.6e  %.6e  %.6e'  ) 
            

