from quantum_env import *
import matplotlib.pyplot as plt

rtype = 'energy'
acttype = 'cont'
# acttype = 'bin'
dt = 0.5
N = 100
P = 6
P_vals = np.arange(10, 1000, 10)
t_vals = P_vals * dt
eres_vals = []
for P in P_vals:
	model = TFIM(N, P, rtype, dt, acttype, g_target = 0)
	# model = SingleSpin(P, rtype, dt, acttype, g_target = 0)
	model.reset()
	for i in range(P):
		gamma = i/P * dt 
		beta =  (1 - i/P) * dt
		action = (gamma, beta)
		model.step(action)
		obs = model.get_observable(model.state)
	energy = -model.get_instantaneous_reward(model.state, P, P, rtype)
	eres = (energy + N) / N
	eres_vals.append(eres)
	print(f"P={P}, E={energy}")

plt.title(f"TFIM (N={N}) ---- Linear digitized-QA ($\\Delta t={dt}$)")
plt.xlabel("$t$")
plt.ylabel("$\\epsilon_{res}$")
plt.yscale("log")
plt.xscale("log")
plt.plot(t_vals, eres_vals)
plt.plot(t_vals, 1/ (np.sqrt(t_vals)), "--", label="$t^{-1/2}$")
plt.legend()
plt.show()