import numpy as np
import quantum_env as qenv
#import quantum as qd
from spinup import ppo
import tensorflow as tf


actType= 'cont'   #bin or cont
model='pSpin' 
tau=np.pi/4 # total time used in the binary action process
P=np.arange(1,3,1)
rew ='energy'  # energy, logE, expE
epochs=100
#paramenters related to the pspin model (for the moment)
N=32
ps=2

for Nt in P:
  dt = tau/Nt
  tf.reset_default_graph() 
  if model == 'SingleSpin':   
    env_fn = lambda : qenv.SingleSpin(Nt,rew,dt,actType)
  elif model == 'pSpin':
    env_fn = lambda : qenv.pSpin(N,ps,Nt,rew,dt,actType)
  else:
    raise ValueError(f'Invalid model:{model}')

  ac_kwargs = dict(hidden_sizes=[16,16], activation=tf.nn.relu)
  dirOut=model+actType+"P"+str(Nt)+"_tau025pi_rw"+rew
  logger_kwargs = dict(output_dir=dirOut, exp_name='RL_first_try')
  ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=1000, epochs=epochs, logger_kwargs=logger_kwargs, gamma=1.0)

