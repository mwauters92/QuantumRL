import numpy as np
import math
import sys
from  scipy.linalg import eigh, toeplitz, det
import matplotlib.pyplot as plt
from IPython import embed
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
    def __init__(self, P, rtype, dt, acttype, N=1, g_target = 0, noise=0, Hx = None, Hz = None):

        # here all variables for you
        self.state = None
        self.m = 0
        self.P = P
        self.rtype=rtype
        self.noise=noise
        self.dt = dt
        self.acttype = acttype
        self.Npart = N        

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

    def set_RL_params(self, acttype, obs_shape, obs_low, obs_high):
        '''
        Sets additional parameters needed by the Reinforment Learning algorithm
        '''
        self.acttype = acttype
        self.obs_shape = obs_shape
        self.obs_low = obs_low
        self.obs_high = obs_high
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=self.obs_shape, dtype=np.float32)
        print('SELF_OBS_SPACE',self.observation_space)
        ## IF CYCLE TO SET ACTION TYPE
        if self.acttype == 'bin':
            self.action_space = spaces.Discrete(2) 
        elif self.acttype == 'cont':
            # TODO issue with action space
            self.action_space = spaces.Box(low= 0., high=2*np.pi, shape=(2,), dtype=np.float32)

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
        return obs.reshape(-1)

    def get_avg_H_target(self, psi):
        H = self.H_target
        E = np.real(np.dot(psi.T.conj(), np.dot(H, psi)))
        return E

    def get_quantum_exect_val(self, Op, psi):
        expect_val = np.real(np.dot(psi.T.conj(), np.dot(Op, psi)))
        return expect_val

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
        self.state = np.copy(self.psi_start)+self.noise*np.random.random(self.psi_start.shape)
        self.state /= np.sqrt(np.vdot(self.state,self.state))
        obs = self.get_observable(self.state)
        return obs #not inter. Now gives back reward = 0

    def get_instantaneous_reward(self, state, m, P,rtype, N=1):
        '''
        From the state computes the instantaneous reward
        Parameters:
            state (complex): system state
            m (int): step during the episode
            P (int): length of the episode
            rtype (string): sets the type of reward given (energy,logE,expE)
            N (int): normalization factor for the energy (default 1)
        Returns:
            reward (real): the reawrd is given only at the end of the episode. It is a function of the system energy at the end of the process. 
        '''
        assert m <= P
        if m == P:
            E = self.get_avg_H_target(self.state)
            if rtype=='energy':
                reward = - E
            elif rtype == 'logE':
                reward = -np.log( 1.e0 - reward + 1.e-8)
            elif rtype == 'expE':
                reward = np.exp(-4*E/N)
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
            assert action in [0, 1]

            # select the correct Hamiltonian
            if action == 0: 
                U = self.np.dot(self.E1, self.U1, self.dt)
            elif action >= 1: 
                U = self.np.dot(self.E2, self.U2, self.dt)
      
            # apply the Hamiltonian evolution
            self.state = np.dot(U, self.state)
        elif self.acttype=='cont':  
            # continuous action below ######## ????? why Hdim ###############################
            action = np.clip(action, 0, 2*np.pi)
 
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
        rewards = self.get_instantaneous_reward(self.state, self.m, self.P, self.rtype, N=self.Npart)
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

    def __init__(self, N, ps, P, rtype, dt, acttype, g_target = 0, noise=0,measured_obs='tomography'):
        
        self.Hx = -self.xSpin(N,0)
        self.Hz = self.pspinHz(N,ps)
        self.N=N
        self.ps=ps
        self.P=P
        self.measured_obs=measured_obs
        self.acttype=acttype
        QuantumEnviroment.__init__(self, P, rtype, dt, acttype, N=N, g_target = 0, noise=0, Hx = self.Hx, Hz = self.Hz)
        self.obs_shape, self.obs_low, self.obs_high = self.get_observable_info()
        self.set_RL_params(self.acttype, self.obs_shape, self.obs_low, self.obs_high)
   
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
            return Sx

    def get_observable(self, state, get_only_info=False):
        if self.measured_obs == "tomography":
            if get_only_info:
                obs_shape = (2 *  self.N+2,)
                obs_low = -1
                obs_high = 1
                return obs_shape, obs_low, obs_high
            state_real = np.real(state)
            state_imag = np.imag(state)
            obs = np.concatenate((state_real, state_imag))

        elif self.measured_obs == "Hobs":
            # get observable shape
            if get_only_info:
                obs_shape = (2,)
                obs_low = -1
                obs_high = 1
                return obs_shape, obs_low, obs_high
            # get averages of Hx and Hz
            avg_Hx = np.vdot(state,np.dot(self.Hx,state)) 
            avg_Hz = np.vdot(state,np.dot(self.Hz,state)) 
            obs = np.array([avg_Hx, avg_Hz])/self.N

        elif self.measured_obs =='sz_distribution':
            if get_only_info:
                obs_shape = (self.N+1,)
                obs_low = 0
                obs_high = 1
                return obs_shape, obs_low, obs_high
            obs=np.abs(state)**2
       
        else:
            raise ValueError(f'Impossible to measure observable:{self.measured_obs} not valid')

        return obs.reshape(-1)


    def get_observable_info(self):
        return self.get_observable(None, get_only_info=True)

        
            

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

#-------------------------------------
# Model class for single spin 1/2
#-------------------------------------
class TFIM(QuantumEnviroment):
    '''Child class of QuantumEnviroment. Add specific model (Transfverse field Ising Model or TFIM) to the class. We use the pseudo-spin picture to decompse the TFIM into a collection of independent two level models. Each model is indicized my a pseudo momenta k. Also see arXiv:1906.08948 .
       Paramenters:
           N (int): number of spin variables
           measured_obs (str): 
            "pesudospin-tomography" ->
            "sx,sz*sz" -> local observables O_x = sigma_x, O_zz = sigma_z*sigma_z

       
       Methods:
           get_full_state: returns the state of the whole ising model
    '''
    def __init__(self, N, P, rtype, dt, acttype, g_target = 0, noise=0, measured_obs="tomography"):
        # initilize model variables
        self.N = N
        self.k = np.pi / self.N * np.arange(1., self.N, 2)
        self.Nk = self.k.shape[0]
        self.state = None
        self.m = 0
        self.P = P
        self.g_target=g_target
        self.rtype = rtype
        self.noise = noise
        self.dt = dt
        self.two_lv_models = []
        self.Hx = []
        self.Hz = []
        self.measured_obs = measured_obs
        self.acttype=acttype
        self.obs_shape, self.obs_low, self.obs_high = self.get_observable_info()
        self.set_RL_params(self.acttype, self.obs_shape, self.obs_low, self.obs_high)

        # define 2 level systems, the equations are taken from arXiv:1906.08948
        for i_k in range(self.Nk):
            k = self.k[i_k]
            Hx = -2 * SIGMA_Z
            Hz = 2 * np.sin(k) * SIGMA_X - 2 * np.cos(k) * SIGMA_Z
            model = QuantumEnviroment(P, rtype, dt, acttype, g_target = g_target, noise=0, Hx = Hx, Hz = Hz)
            self.two_lv_models.append(model)
            self.Hx.append(Hx)
            self.Hz.append(Hz)


    def get_full_state(self):
        two_lv_states = [model.state for model in self.two_lv_models]
        return np.array(two_lv_states)

    def get_avg_H_target(self, state):
        E = 0.
        for i_k in range(self.Nk):
            two_lv_state = state[i_k]
            two_lv_model = self.two_lv_models[i_k]
            E = E + two_lv_model.get_avg_H_target(two_lv_state)
        return E

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
        for model in self.two_lv_models: 
            model.step(action)
        # this part is the same for both binary and blabla
        self.state = self.get_full_state()
        obs = self.get_observable(self.state)
        self.m += 1
        if self.m == self.P: done = True
        else: done = False
        rewards = self.get_instantaneous_reward(self.state, self.m, self.P, self.rtype, self.N)
        return np.array(obs), rewards, done, {}

    def reset(self):
        '''
        Reset the state to the initial one, with the possible addition of disorder, set by self.noise
        Returns: 
            obs (real): observable of reset state
        '''
        for model in self.two_lv_models:
            model.reset()
        self.m = 0
        self.state = self.get_full_state()
        obs = self.get_observable(self.state)
        return obs #not inter. Now gives back reward = 0
 
    def get_observable(self, state, get_only_info=False):
        if self.measured_obs == "tomography":
            if get_only_info: 
                obs_low=-1
                obs_high=1
                obs_shape = (2 * 2 * self.Nk,)
                return obs_shape, obs_low, obs_high
            state_real = np.real(state)
            state_imag = np.imag(state)
            obs = np.concatenate((state_real, state_imag))

        elif self.measured_obs == "Hobs":
            # get observable shape
            if get_only_info:
                obs_shape = (2 * self.N,)
                obs_low=-1
                obs_high=1
                obs_shape = (2,) #delete in future
                return obs_shape, obs_low, obs_high
            # get averages of Hx and Hz
            avg_Hx = 0.
            avg_Hz = 0.
            for i_k in range(self.Nk):
                two_lv_state = state[i_k]
                two_lv_model = self.two_lv_models[i_k]
                #print('i_k=',i_k,' Nk=', self.Nk, ' lenHx=',len(self.Hx), ' lenHz=',len(self.Hz))
                Hx_k = self.Hx[i_k]
                Hz_k = self.Hz[i_k]
                avg_Hx = avg_Hx + two_lv_model.get_quantum_exect_val(Hx_k, two_lv_state)
                avg_Hz = avg_Hz + two_lv_model.get_quantum_exect_val(Hz_k, two_lv_state)
            
            # get  averages 
            avg_sum_sx = -avg_Hx
            avg_sum_szsz = avg_Hz

            # get local observables for each site
            obs_x = (avg_sum_sx / self.N) * np.ones(self.N)
            obs_zz = (avg_sum_szsz / self.N) * np.ones(self.N)
            obs = np.concatenate([obs_x, obs_zz])
            obs = np.array([avg_sum_sx, avg_sum_szsz])/self.N #delete in future  

        elif self.measured_obs == "HCorr":
            # get observable shape
            if get_only_info:
                obs_low=-1
                obs_high=1
                obs_shape = (3,) #delete in future
                return obs_shape, obs_low, obs_high
            # get averages of Hx and Hz
            avg_Hx = 0.
            avg_Hz = 0.
            for i_k in range(self.Nk):
                two_lv_state = state[i_k]
                two_lv_model = self.two_lv_models[i_k]
                #print('i_k=',i_k,' Nk=', self.Nk, ' lenHx=',len(self.Hx), ' lenHz=',len(self.Hz))
                Hx_k = self.Hx[i_k]
                Hz_k = self.Hz[i_k]
                avg_Hx = avg_Hx + two_lv_model.get_quantum_exect_val(Hx_k, two_lv_state)
                avg_Hz = avg_Hz + two_lv_model.get_quantum_exect_val(Hz_k, two_lv_state)
            
            # get  averages 
            avg_sum_sx = -avg_Hx
            avg_sum_szsz = avg_Hz
            avg_sum_corrZ_2 = self.get_correlation(state,2)*self.N
            # get local observables for each site
            obs = np.array([avg_sum_sx, avg_sum_szsz, avg_sum_corrZ_2])/self.N #delete in future  

        else:
            raise ValueError(f'Impossible to measure observable:{self.measured_obs} not valid')

        return obs.reshape(-1)
    
    def get_observable_info(self):
        return self.get_observable(None, get_only_info=True)

    def get_correlation(self, state, n):
        """
        Computes the correlation at distance n on the TFIM model
        """
        corr_mat=np.zeros([n,n])
        corr_row=[]
        corr_col=[0]
        for i_n in np.arange(0,n):
            Gn=0.
            Gn_m=0.
            for i_k in range(self.Nk):
                k=self.k[i_k]
                two_lv_state = state[i_k]
                two_lv_model = self.two_lv_models[i_k]
                Gzk = 2 * np.sin((i_n+1)*k) * SIGMA_X - 2 * np.cos((i_n+1)*k) * SIGMA_Z
                Gzk_m = -2 * np.sin(i_n*k) * SIGMA_X - 2 * np.cos(i_n*k) * SIGMA_Z
                Gn = Gn + two_lv_model.get_quantum_exect_val(Gzk, two_lv_state)
                Gn_m = Gn_m + two_lv_model.get_quantum_exect_val(Gzk_m, two_lv_state)
            '''
            for j_n in range(n-i_n):
                corr_mat[j_n,j_n+i_n]=Gn /self.N
                if j_n+i_n+1 < n : corr_mat[j_n+i_n+1,j_n+i_n]=Gn_m /self.N
            '''
            corr_row.append(Gn / self.N)
            corr_col.append(Gn_m / self.N)
        corr_col[0]=corr_row[0]
        corr_col.pop(-1)
        corr_mat=toeplitz(corr_col, corr_row)
        return det(corr_mat)


