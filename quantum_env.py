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

#--------------------------------------
# General quantum enviroment for Reinforcement Learning
#--------------------------------------
class QuantumEnviroment():
    '''
    General class that is passed to Spinningup reinforcment learning routines
    Variables:
        P (int): number of steps in one episode
        rtype (string): type of reward computed from the system energy E at the end of the episode 
            - energy: the reward is -E 
            - expE: reward is exp(-2*E)
            - logE: reward is -log( 1 + E + 1.e-8) this can be used only if the minimum energy (rescaled) is known 
        dt (float32): evolution time at each step of an episode. It is used onyly if the action in binary
        acttype (string): set the type of actions that can be performed on the system
            - bin: at each step the system evolve with H1 or H2 ONLY, for a time step dt
            - cont: at each step the system evolves with both H1 and H2; the action consits in choosing the two time steps
        g_target (float32): transverse field in the target Hamiltonian H_target = H1 + g_target * H2. Default is 0
        noise (float32): noise added to the initial state at the beginning of each episode. Default is 0
    

    Methods:
        __init__: constructor
        reset: reset the state to inital one + noise
        get_observable: from the quantum state gets the observable passed to the RL algorithm
        get_dense_Uevol: perform quantum unitary evolution on the state  
        get_instantaneous_reward: computes the reard given the state and the step number inside an episode
        step: perform a single action on the state
        render: does nothing but it is required by the RL algorithm
    Please notice that all methods but get_dense_Uevol are necessary for spinningup
    '''
    def __init__(self, P, rtype, dt, acttype, g_target = 0, noise=0, Hx = None, Hz = None):

        # here all variables for you
        self.state = None
        self.m = 0
        self.P = P
        self.rtype=rtype
        self.noise=noise
        self.dt = dt
        self.acttype = acttype

        self.H1 = Hz
        self.H2 = Hx
        self.H_target = Hz + g_target * Hx
        self.H_start = Hx

        self.Hdim = Hz.shape[0]
        self.obs_shape = (2*self.Hdim,)  
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=self.obs_shape, dtype=np.float32)
        ## IF CYCLE TO SET ACTION TYPE
        if self.acttype == 'bin':
            self.action_space = spaces.Discrete(2) 
        elif self.acttype == 'cont':
            # TODO issue with action space
            self.action_space = spaces.Box(low= 0., high=2*np.pi, shape=(2,), dtype=np.float32)
        else:
            raise ValueError(f'Wrong action type:{self.acttype} not valid')
        # the action space has to be defined somewhere !!!
        
        # get initial state
        E0, eigstates0 = eigh(self.H_start)            
        self.psi_start = eigstates0[:,0]
        self.E1, self.U1 = eigh(self.H1)
        self.E2, self.U2 = eigh(self.H2)

    # FUNCTION THAT I NEED FOR RL

    def get_observable(self, state):
        '''
        It computes the observable given to the RL algorithm. 
        In the actual implementation it is just the full state stored in 
        a real vector.
        Parameters:
            state (complex): complex vector of dmnesion self.Hdim describing the system state
        Returns:
        '''     
        state_real = np.real(state)
        state_imag = np.imag(state)
        obs = np.concatenate((state_real, state_imag))
        return obs


    def get_dense_Uevol(self, E, U, dt):
        '''
        Computes the unitary time evolution operator
        Parameters:
            E (real): eigenvalues of of the Hamniltonian that enters the evolution operator
            U (complex): matrix with the eigenvectors of the hamiltonian
            dt (real): time step
        Returns:
            exp_H (complex): time evolution operator
        '''
        exp_E = np.exp(-1j * E * dt)
        exp_H = np.dot(U, np.multiply(exp_E[:,None], U.conj().T))
        return exp_H


    def reset(self):
        '''
        Reset the state to the initial one, with the possible addition of disorder, set by self.noise
        Returns: 
            obs (real): observable of reset state
        '''
        # initialize a new simulation:
        # I will call it after receiving a flag "done"
        self.m = 0
        self.state = np.copy(self.psi_start)+self.noise*np.random.rand(self.Hdim)
        self.state/=np.sqrt(np.vdot(self.state,self.state))
        obs = self.get_observable(self.state)
        return obs #not inter. Now gives back reward = 0


    def get_instantaneous_reward(self, state, m, P,rtype):
        '''
        From the state computes the instantaneous reward
        Parameters:
            state (complex): system state
            m (int): step during the episode
            P (int): length of the episode
            rtype (string): sets the type of reward given (energy,logE,expE)
        Returns:
            reward (real): the reawrd is given only at the end of the episode. It is a function of the system energy at the end of the process. 
        '''
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
                raise ValueError(f"wrong reward type:{rtype} is not a valid rtype")
        else: reward = 0
        return reward


    def step(self, action):
        '''
         Update the state according to the action chosen by the RL algorithm
         Parameters:
             action (real): action to be performed on the system. depends on actType
         Returns:
             obs (real): updated observable of the state
             reward (real): reward of the updated state
             done (bool): if True the episode is over.
        '''
        # here I want the quantum evolution for 1 step, 
        # given the current quantum internal state self.state and the action in input "action"
        # action == 0 -> evolve with H1
        # action == 1 -> evolve with H2
        if self.acttype=='bin':
            #assert action in [0, 1]

            # select the correct Hamiltonian
            if action == 0: 
                U = self.get_dense_Uevol(self.E1, self.U1, self.dt)
            elif action >= 1: 
                U = self.get_dense_Uevol(self.E2, self.U2, self.dt)
      
            # apply the Hamiltonian evolution
            self.state = np.dot(U, self.state)
        elif self.acttype=='cont':  
            # continuous action below
            action = np.clip(action, 0, np.pi*self.Hdim)
 
            # apply the Hamiltonian evolution

            U = self.get_dense_Uevol(self.E1, self.U1, action[0])
            self.state = np.dot(U, self.state)
            U = self.get_dense_Uevol(self.E2, self.U2, action[1])
            self.state = np.dot(U, self.state)
 
        else:
            raise ValueError(f'Wrong action type:{self.acttype} not valid')

        # this part is the same for both binary and blabla
        obs = self.get_observable(self.state)
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
# End of class QuantumEnviroment
# ------------------------------------


#-------------------------------------
# Model class for pSpin model
#-------------------------------------

class pSpin(QuantumEnviroment):
    '''Children class of QuantumEnviroment. Add specific model (pspin) to the class.
       Paramenters:
           N (int): number of spin variables
           ps (int): rank of the interaction
       
       Methods:
           pspinHz: construct the z-part of the Hamiltonian (diagonal)
           xSpin: contruct the  x-component of the global spin operator 
    '''

    def __init__(self, N, ps, P, rtype, dt, acttype, g_target = 0, noise=0):
        
        Hx = -self.xSpin(N,0)
        Hz = self.pspinHz(N,ps)
        QuantumEnviroment.__init__(self, P, rtype, dt, acttype, g_target = 0, noise=0, Hx = Hx, Hz = Hz)
   
    def pspinHz(self,N,p):
        """Construct the x and z Hamiltonian fo the pspin model
           Parameters:
               N (int): chain length
               p (int): rank ofthe interaction
           Returns:
               Hz (real): interaction part of the Hamiltonian of the pSpin model
        """
        Hz=np.zeros([N+1,N+1])
        x=np.linspace(-1,1,N+1,endpoint=True);
        Hz=-N*np.diag(x**p);
        return Hz

    def xSpin(self,N,job):
        """Transverse field S_x in the z basis representation.
           Parameters:
               N (int): chain length
               job (int): 0 or 1 tells if to diagonalize Sx
           Returns:
               Sx (real): x component of the total spin
               Ux (complex): if job=1 Ux contains the eigenvectors of Sx
        """
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
#------------------------------------------------------------
# End of pSpin class
#-----------------------------------------------------------

#-------------------------------------
# Model class for single spin 1/2
#-------------------------------------

class SingleSpin(QuantumEnviroment):
    '''Children class of QuantumEnviroment. Add specific model (single spin 1/2) to the class.
       It passes to QuantumEnviroment only the pauli matrices
    '''

    def __init__(self, P, rtype, dt, acttype, g_target = 0, noise=0):
        
        QuantumEnviroment.__init__(self, P, rtype, dt, acttype, g_target = 0, noise=0, Hx = -np.copy(SIGMA_X), Hz = -np.copy(SIGMA_Z) )
   

#---------------------------------------
# end of class SingleSpin
# ------------------------------------


