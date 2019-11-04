import numpy as np
import math
import sys
from  scipy.linalg import eigh
import matplotlib.pyplot as plt
from gym import spaces

# ---------------------------------------

# ---------------------------------------
# EQUIVALENT OF GYM FOR QUANTUM ENVIRONMENT
# ---------------------------------------


#===============================================================================
# class QUANTUM(gym.Env):
#     metadata = {'render.modes': ['human']} # what is this?
# 
#     def __init__(self):
# 
#   def step(self, action):
#     ...
#   def reset(self):
#     ...
#   def render(self, mode='human'):
#     ...
#   def close(self):
#     ...
#===============================================================================



# ---------------------------------------
# Constants
# ---------------------------------------

SIGMA_0 = np.array([[1, 0],[0, 1]])
SIGMA_X = np.array([[0, 1],[1, 0]])
SIGMA_Y = np.array([[0, -1j],[1j, 0]])
SIGMA_Z = np.array([[1, 0],[0, -1]])

# ------------------------------------
# class Spin_cont
# ------------------------------------


class Spin_cont():

    def __init__(self, P,rtype,  g_target = 0, noise=0, Hx = -np.copy(SIGMA_X), Hz = -np.copy(SIGMA_Z)):

        # here all variables for you
        self.state = None
        self.m = 0
        self.P = P
        self.rtype=rtype
        self.noise=noise
        self.H1 = Hz
        self.H2 = Hx
        self.H_target = Hz + g_target * Hx
        self.H_start = Hx
        self.Hdim = Hz.shape[0]

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=0., high=2*np.pi, shape=(2,), dtype=np.float32)

        # get initial state
        E0, eigstates0 = eigh(self.H_start)            
        self.psi_start = eigstates0[:,0]


# --------------------------

# FUNCTION THAT I NEED FOR RL

    def description_of_state(self, state):
        state_real = np.real(state)
        state_imag = np.imag(state)
        obs = np.concatenate((state_real, state_imag))
        return obs
 

    def get_dense_Uevol(self, H, dt):
        E, U = eigh(H)
        exp_E = np.exp(-1j * E * dt)
        exp_H = np.dot(U, np.multiply(exp_E[:,None], U.conj().T))
        return exp_H


    def reset(self):
        # initialize a new simulation:
        # I will call it after receiving a flag "done"
        self.m = 0
        self.state = np.copy(self.psi_start)+self.noise*np.random.rand(self.Hdim)
        self.state/=np.sqrt(np.vdot(self.state,self.state))
        obs = self.description_of_state(self.state)
        return obs #not inter. Now gives back reward = 0


    def get_instantaneous_reward(self, state, m, P, rtype, a):   # a is action as given in step [0, 2pi]
        assert m <= P
        if m == P:
            E = np.real(np.dot(self.state.T.conj(), np.dot(self.H_target, self.state)))
            if rtype == "energy":
                reward = - E
            elif rtype == "logE":
                reward = -np.log( 1.e0 - reward + 1.e-8)
            elif rtype == "expE":
                reward = np.exp(-2*E)
            else:
                reward=0
                print('wrong rtype')
            #momo.write(str(-E)+'\n')
        else: 
            reward =  0 #(0-np.any(a>0))*0.01
        return reward



    def step(self, action):
        # here I want the quantum evolution for 1 step, 
        # given the current quantum internal state self.state and the action in input "action"
        
        action = np.clip(action, 0, 2*np.pi)
 
        # apply the Hamiltonian evolution

        U = self.get_dense_Uevol(self.H1, action[0])
        self.state = np.dot(U, self.state)
        U = self.get_dense_Uevol(self.H2, action[1])
        self.state = np.dot(U, self.state)
        obs = self.description_of_state(self.state)
        self.m += 1
        if self.m == self.P: done = True
        else: done = False
        rewards = self.get_instantaneous_reward(self.state, self.m, self.P, self.rtype, action)
        return np.array(obs), rewards, done, {}

    def close(self):
        pass

    def render(self):
        pass  


# ------------------------------------
# End of class Spin_cont
# ------------------------------------


class Spin_bin():

    def __init__(self, P, dt, rtype, g_target = 0, noise=0, Hx = -np.copy(SIGMA_X), Hz = -np.copy(SIGMA_Z)):

        # here all variables for you
        self.state = None
        self.m = 0
        self.P = P
        self.dt = dt
        self.rtype=rtype
        self.noise=noise
        self.H1 = Hz
        self.H2 = Hx
        self.H_target = Hz + g_target * Hx
        self.H_start = Hx
        self.Hdim = Hz.shape[0]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        # get initial state
        E0, eigstates0 = eigh(self.H_start)            
        self.psi_start = eigstates0[:,0]
        #self.psi_start = np.array([1, 0])



# --------------------------

# FUNCTION THAT I NEED FOR RL

    def description_of_state(self, state):
        state_real = np.real(state)
        state_imag = np.imag(state)
        obs = np.concatenate((state_real, state_imag))
        return obs

 

    def get_dense_Uevol(self, H, dt):
        E, U = eigh(H)
        exp_E = np.exp(-1j * E * dt)
        exp_H = np.dot(U, np.multiply(exp_E[:,None], U.conj().T))
        return exp_H



    def reset(self):
        # initialize a new simulation:
        # I will call it after receiving a flag "done"
        self.m = 0
        self.state = np.copy(self.psi_start)+self.noise*np.random.rand(self.Hdim)
        self.state/=np.sqrt(np.vdot(self.state,self.state))
        obs = self.description_of_state(self.state)
        return obs #not inter. Now gives back reward = 0


    def get_instantaneous_reward(self, state, m, P,rtype):
        assert m <= P
        if m == P:
            E = np.real(np.dot(self.state.T.conj(), np.dot(self.H_target, self.state)))
            if rtype=='energy':
                reward = - E
            elif rtype == 'logE':
                reward = -np.log( 1.e0 - reward + 1.e-8)
            elif rtype == 'expE':
                reward = np.exp(-2*E)
            else: 
                reward = 0
                print("wrong rtype")
        else: reward = 0
        return reward



    def step(self, action):
        # here I want the quantum evolution for 1 step, 
        # given the current quantum internal state self.state and the action in input "action"
        # action == 0 -> evolve with H1
        # action == 1 -> evolve with H2

        assert action in [0, 1]

        # select the correct Hamiltonian
        if action == 0: H = self.H1
        elif action == 1: H = self.H2

        # apply the Hamiltonian evolution

        U = self.get_dense_Uevol(H, self.dt)
        self.state = np.dot(U, self.state)
        obs = self.description_of_state(self.state)
        self.m += 1
        if self.m == self.P: done = True
        else: done = False
        rewards = self.get_instantaneous_reward(self.state, self.m, self.P, self.rtype)
        return np.array(obs), rewards, done, {}

    def close(self):
        pass

    def render(self):
        pass  



# ------------------------------------
# End of class Spin_bin
# ------------------------------------

class Pspin_cont():

    def __init__(self, N, p, P,rtype,  g_target = 0, noise=0):

        # here all variables for you
        self.state = None
        self.m = 0
        self.N = N
        self.p = p
        self.P = P
        self.rtype=rtype
        self.noise=noise
        self.Hz = self.pspinHz(N,p)
        self.Hx , self.Ux = self.S_x(N,1)
        self.H_target = self.Hz - g_target * self.Hx
        self.Hdim = N+1

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2*self.Hdim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0., high=2*np.pi, shape=(2,), dtype=np.float32)

        # get initial state
        E0, eigstates0 = eigh(-self.Hx)            
        self.psi_start = eigstates0[:,0]


# --------------------------

# FUNCTION THAT I NEED FOR RL

    def pspinHz(self,N,p):
        """Construct the x and z Hamiltonian fo the pspin model"""
        Hz=np.zeros([N+1,N+1])
        x=np.linspace(-1,1,N+1,endpoint=True);
        Hz=-N*np.diag(x**p);
        return Hz

    def S_x(self,N,job):
        """Transverse field S_x in the z basis representation.
        if job==1 it provides also the unitary matrix that 
        diagonalizes S_x"""
        Sx=np.zeros([N+1,N+1])
        x=np.linspace(-1,1,N+1);
        for j in range(N):
            Sx[j+1,j]=0.5*N*np.sqrt( (1.-x[j])*(1.+x[j]+2.0/N) );
        Sx+=Sx.transpose();
        if (job==1):
            _,Usx=eigh(Sx)
            return Sx, Usx
        elif(job==0):
            return(Sx)


    def description_of_state(self, state):
        state_real = np.real(state)
        state_imag = np.imag(state)
        obs = np.concatenate((state_real, state_imag))
        return obs
 

    def get_dense_UevolZ(self, H, dt):
        exp_H = np.exp(-1j * H.diagonal() * dt)
        return np.diag(exp_H)

    def get_dense_UevolX(self, U,dt):
        N=self.Hdim-1
        exp_E=np.exp(1j*dt*np.linspace(-N,N,N+1))
        exp_H = np.dot(U, np.multiply(exp_E[:,None], U.conj().T))
        return exp_H

    def reset(self):
        # initialize a new simulation:
        # I will call it after receiving a flag "done"
        self.m = 0
        self.state = np.copy(self.psi_start)+self.noise*np.random.rand(self.Hdim)
        self.state/=np.sqrt(np.vdot(self.state,self.state))
        obs = self.description_of_state(self.state)
        return obs #not inter. Now gives back reward = 0


    def get_instantaneous_reward(self, state, m, P, rtype, a):   # a is action as given in step [0, 2pi]
        assert m <= P
        if m == P:
            E = np.real(np.dot(self.state.T.conj(), np.dot(self.H_target, self.state)))
            if rtype == "energy":
                reward = - E
            elif rtype == "logE":
                reward = -np.log( 1.e0 - reward + 1.e-8)
            elif rtype == "expE":
                reward = np.exp(-2*E)
            else:
                reward=0
                print('wrong rtype')
            #momo.write(str(-E)+'\n')
        else: 
            reward =  0 #(0-np.any(a>0))*0.01
        return reward



    def step(self, action):
        # here I want the quantum evolution for 1 step, 
        # given the current quantum internal state self.state and the action in input "action"
        
        action = np.clip(action, 0, np.pi*self.Hdim)
 
        # apply the Hamiltonian evolution

        U = self.get_dense_UevolZ(self.Hz, action[0])
        self.state = np.dot(U, self.state)
        U = self.get_dense_UevolX(self.Ux, action[1])
        self.state = np.dot(U, self.state)
        obs = self.description_of_state(self.state)
        self.m += 1
        if self.m == self.P: done = True
        else: done = False
        rewards = self.get_instantaneous_reward(self.state, self.m, self.P, self.rtype, action)
        return np.array(obs), rewards, done, {}

    def close(self):
        pass

    def render(self):
        pass  


# ------------------------------------
# End of class Pspin_cont
# ------------------------------------
