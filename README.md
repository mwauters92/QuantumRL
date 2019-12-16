# QuantumRL
classes and scripts for ground state preparation with reinforcement learning. It requires a running version of OpenAI spinningup libray. 
Installation instruction can be find in https://spinningup.openai.com/en/latest/user/installation.html

Files
quantum_env.py 
  library that implement the classes needed for the RL algorithm.

RunningRL.py
  set up and train two neural networks in the actor-critic framework.
  by default if trains an RL agent on the single spin 1/2 with continuous actions, if no in-line options are give.
  type python RunningRL.py --help to see all possible arguments to pass to the program
  
test_quantum.py(obsolete) 
  run a trained version of the network. It needs the directory where the model has been saved.
  It writes s file with the actions perfomed during several episodes (actions.dat) and a summary file with the average performances (summary.dat)
