import numpy as np
import quantum_env as qenv
#import quantum as qd
from spinup import ppo
import tensorflow as tf

actType= 'pspin'   #bin or cont
tau=np.pi/4
P=np.concatenate((np.arange(2,20,2),np.arange(20,64,4)))
P=np.arange(1,18,1)
#P=[1]
rew ='energy'  # energy, logE, expE
epochs=1000

if actType == 'bin':
  for Nt in P:
    dt = tau/Nt
    tf.reset_default_graph()    
    env_fn = lambda : qenv.Spin_bin(Nt,dt,rtype=rew,noise=0.2)
    ac_kwargs = dict(hidden_sizes=[16,16], activation=tf.nn.relu)
    dirOut=actType+"P"+str(Nt)+"_tau025pi_rw"+rew+'_noise0.2'
    logger_kwargs = dict(output_dir=dirOut, exp_name='RL_first_try')
    ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=1000, epochs=epochs, logger_kwargs=logger_kwargs, gamma=1.0)

#actType='cont'
if actType == 'cont':
  for Nt in P:
    tf.reset_default_graph()    
    env_fn = lambda : qenv.Spin_cont(Nt,rtype=rew,noise=1.0)
    #env_fn = lambda : qc.Quantum(Nt,rtype=rew)
    ac_kwargs = dict(hidden_sizes=[16,16], activation=tf.nn.relu)
    dirOut=actType+"P"+str(Nt)+"_rw"+rew+'_noise1.0'
    logger_kwargs = dict(output_dir=dirOut, exp_name='RL_first_try')
    ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=1000, epochs=epochs, logger_kwargs=logger_kwargs, gamma=1.0)



if actType == 'pspin':
  N=32
  for Nt in P:
    tf.reset_default_graph()    
    env_fn = lambda : qenv.Pspin_cont(N,2,Nt,rtype=rew)
    #env_fn = lambda : qc.Quantum(Nt,rtype=rew)
    ac_kwargs = dict(hidden_sizes=[16,16], activation=tf.nn.relu)
    dirOut=actType+"P"+str(Nt)+"_N"+str(N)+"_rw"+rew
    logger_kwargs = dict(output_dir=dirOut, exp_name='RL_first_try')
    ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=1000, epochs=epochs, logger_kwargs=logger_kwargs, gamma=1.0)

