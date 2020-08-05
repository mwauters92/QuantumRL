# QuantumRL
classes and scripts for ground state preparation with reinforcement learning. It requires a running version of OpenAI spinningup libray. 
Installation instruction can be find in https://spinningup.openai.com/en/latest/user/installation.html
WARNING: the new version of spinningup requires both tensorflow and pytorch. Both are installed along with spinningup, but there can be issues with the tmp directory.
An installation of mpi is needed, otherwise spinningup will not compile. 

Files
quantum_env.py 
  library that implement the classes needed for the RL algorithm.
  models:
  pSpin: ferromagnetic p-spin ising model
  SingleSpin: single spin 1/2 in magnetic field
  TFIM: transverse field Ising model
  RandomTFIM: TFIM with random couplings
  SKglass: Sherrington-KIrkpatrick spin glass, a fully-connected Ising model with coupling randomly +/-1 

RunningRL.py
  set up and train two neural networks in the actor-critic framework.
  by default if trains an RL agent on the single spin 1/2 with continuous actions, if no in-line options are give.
  type python RunningRL.py --help to see all possible arguments to pass to the program
  
test_quantum.py 
  run a trained version of the network. It needs the directory where the model has been saved.
  It writes a file with the actions perfomed during several episodes (actions.dat) and a summary file with the average performances (summary.dat)
  It has the same default options of RunningRL.py, plus an extra arguments that sets the number of validation steps.
  Local optimization can be performed on top of RL actions.

test_twoSize.py 
  tests the policy transferability. It needs a trained network for a system of size N_act, then saves a file summary_rew.dat in the folder corresponding to N_act with the policy tested on a system of size N_rew.

