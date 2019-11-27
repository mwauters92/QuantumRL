import numpy as np
import quantum_env as qenv
#import quantum as qd
from spinup import ppo
import tensorflow as tf


actType= 'cont'                 # action type: bin, cont
model='pSpin'                   # model : pSpin, SingleSpin, TFIM
tau=20                          # total time used in the binary action process
P=np.arange(2,19,2)             # list of episode lenghts 
P=[8]
rtype ='energy'                 #reward types: energy, logE, expE
epochs=256                      # number of epochs
nstep=8192                      # steps per episodes

# physical parameters
N=np.array([16,32,64,128])      # list of sizes
N=[32]
ps=2                            # interaction rank of the pSpin model

dirO = "../Output/"+model+"/ep"+str(epochs)+"_sep"+str(nstep)+"/"
for Nt in P:
  dt = tau/Nt
  for Ns in N:
    tf.reset_default_graph() 
    if model == 'SingleSpin':   
      env_fn = lambda : qenv.SingleSpin(Nt,rtype,dt,actType)
      dirOut=dirO+model+actType+"P"+str(Nt)+'_rw'+rtype
    elif model == 'pSpin':
      env_fn = lambda : qenv.pSpin(Ns,ps,Nt,rtype,dt,actType)
      dirOut=dirO+'pspin'+"P"+str(Nt)+'_N'+str(Ns)+'_rw'+rtype
    elif model == 'TFIM':
      env_fn = lambda : qenv.TFIM(Ns,Nt,rtype,dt,actType)
      dirOut=dirO+'TFIM'+"P"+str(Nt)+'_N'+str(Ns)+'_rw'+rtype
    else:
      raise ValueError(f'Invalid model:{model}')
    dirOut += '/network32x16'
    ac_kwargs = dict(hidden_sizes=[32,16], activation=tf.nn.relu)
    logger_kwargs = dict(output_dir=dirOut, exp_name='RL_first_try')
    ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=nstep, epochs=epochs, logger_kwargs=logger_kwargs, gamma=1.0)

