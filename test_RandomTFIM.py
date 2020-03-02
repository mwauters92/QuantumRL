from quantum_env import *
import matplotlib.pyplot as plt

rtype = 'energy'
acttype = 'cont'
# acttype = 'bin'
dt = 0.1
N = 50
J = np.random.rand(N)
J[-1] = 0.
J = np.ones(N)

P_vals = np.arange(10, 1000, 10)
t_vals = P_vals * dt
g_target = 0.

Emin = - np.sum(np.abs(J))
Emax = + np.sum(np.abs(J))
fmt = ["o-", ".-"]
for i_model, model_opt in enumerate(["RandomTFIM", "TFIM"]):
	eres_vals = []
	for P in P_vals:
		if model_opt == "RandomTFIM": 
			model = RandomTFIM(N, J, P, rtype, dt, acttype, g_target = g_target)
		elif model_opt == "TFIM": 
			model = TFIM(N, P, rtype, dt, acttype, g_target = g_target)

		# model = SingleSpin(P, rtype, dt, acttype, g_target = 0)
		model.reset()
		for i in range(P):
			gamma = i/P * dt 
			beta =  (1 - i/P) * dt
			action = (gamma, beta)
			model.step(action)
			obs = model.get_observable(model.state)
		energy = -model.get_instantaneous_reward(model.state, P, P, rtype)
		eres = (energy - Emin) / (Emax - Emin)
		eres_vals.append(eres)
		print(f"P={P}, Emin={Emin}, E={energy}")

	plt.plot(t_vals, eres_vals, fmt[i_model], label=model_opt)

plt.plot(t_vals, 1/ (np.sqrt(t_vals)), "--", label="$t^{-1/2}$")
plt.title(f"Translationally invariant Ising chain (N={N}) ---- Linear digitized-QA ($\\Delta t={dt}$)")
plt.xlabel("$t$")
plt.ylabel("$\\epsilon_{res}$")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.show()