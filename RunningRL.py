import numpy as np
import quantum_env as qenv
#import quantum as qd
from spinup import ppo_tf1 as ppo
import tensorflow as tf
import argparse
import time

parser = argparse.ArgumentParser(description="Reinforcement Learing for QAOA")
parser.add_argument('--model', default="SingleSpin", type=str, help="Model to study with QAOA+RL")
parser.add_argument('--obs', default="tomography", type=str, help="observables seen by the Reinforcement Learing agent")
parser.add_argument('--actType', default="cont", type=str, help="action space of the RL agent: cont or bin")
parser.add_argument('--tau', default=10, type=float, help="total time for binary actions")
parser.add_argument('--Pvars', default=[2,3,1], nargs=3, type=int, help="episodes lengths (start,stop,step)")
parser.add_argument('--network', default=[16,16],nargs='+', type=int, help="Number of neurons in each layer")
parser.add_argument('--rtype', default="energy", type=str, help="reward function (energy, expE or logE")
parser.add_argument('--epochs', default=512, type=int, help="Number of epochs of the training process")
parser.add_argument('--nstep', default=1024, type=int, help="Number of steps per epoch")
parser.add_argument('--N', default=32, type=int, help="System size (only for pSpin and TFIM)")
parser.add_argument('--ps', default=2, type=int, help="Interaction rank (only for pSpin)")
parser.add_argument('--hfield', default=0, type=float, help="Transverse field for target state")
parser.add_argument('--noise', default=0, type=float, help="Noise on the initial state")
parser.add_argument('--seed', default = 812453, type=int, help="Seed for the RandomTFIM model")
args = parser.parse_args()

actType=args.actType                 # action type: bin, cont
model=args.model                   # model : pSpin, SingleSpin, TFIM
tau=args.tau                          # total time used in the binary action process
P=np.arange(args.Pvars[0],args.Pvars[1],args.Pvars[2])             # list of episode lenghts 
#P=[8]
measured_obs = args.obs
rtype =args.rtype                 #reward types: energy, logE, expE
epochs=args.epochs                      # number of epochs
nstep=args.nstep                      # steps per episodes
layers=args.network

# physical parameters
N=[args.N]
ps=args.ps                      # interaction rank of the pSpin model
hfield = args.hfield
noise=args.noise
seed = args.seed

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
  dt = tau/Nt
  for Ns in N:
    #tf.reset_default_graph() 
    tf.compat.v1.reset_default_graph()
    if model == 'SingleSpin':   
      env_fn = lambda : qenv.SingleSpin(Nt,rtype,dt,actType, noise=noise, g_target=hfield)
      dirOut=dirO+model+actType+"P"+str(Nt)+'_rw'+rtype
    elif model == 'pSpin':
      env_fn = lambda : qenv.pSpin(Ns,ps,Nt,rtype,dt,actType,measured_obs=measured_obs, g_target=hfield, noise=noise)
      dirOut=dirO+'pspin'+"P"+str(Nt)+'_N'+str(Ns)+'_rw'+rtype
    elif model == 'TFIM':
      env_fn = lambda : qenv.TFIM(Ns,Nt,rtype,dt,actType,measured_obs=measured_obs, g_target=hfield, noise=noise)
      dirOut=dirO+'TFIM'+"P"+str(Nt)+'_N'+str(Ns)+'_rw'+rtype
    elif model == 'RandomTFIM':
      J_couplings = set_couplings(Ns, seed)
      env_fn = lambda : qenv.RandomTFIM(Ns,J_couplings,Nt,rtype,dt,actType,measured_obs=measured_obs, g_target=hfield, noise=noise)
      dirOut=dirO+'RandomIsing'+"P"+str(Nt)+'_N'+str(Ns)+'_rw'+rtype
    else:
      raise ValueError(f'Invalid model:{model}')
    dirOut += '/'+measured_obs+'/network'+str(layers[0])+'x'+str(layers[1])
    ac_kwargs = dict(hidden_sizes=layers, activation=tf.nn.relu)
    logger_kwargs = dict(output_dir=dirOut, exp_name='RL_first_try')
    ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=nstep, epochs=epochs, logger_kwargs=logger_kwargs, gamma=1.0,target_kl=0.01, save_freq=128)
