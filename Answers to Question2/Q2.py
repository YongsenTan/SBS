import numpy as np
import matplotlib.pyplot as plt

# Define the rate constants
k1 = 100.0  # /uM/min
k2 = 600.0  # /uM/min
k3 = 150.0  # /uM/min

# Define the initial conditions
E0 = 1.0  # uM
S0 = 10.0  # uM
ES0 = 0.0  # uM
P0 = 0.0  # uM

# Define the time step and simulation time
dt = 0.00005  # min
tmax = 0.02  # min


# Define the function that computes the derivatives of the dependent variables
def derivs(S, E, ES, P):
    v1 = k1 * E * S - k2 * ES
    v2 = k3 * ES
    dSdt = -v1
    dEdt = -v1
    dESdt = v1 - v2
    dPdt = v2
    return dSdt, dEdt, dESdt, dPdt


# Initialize the dependent variables
S = np.zeros(int(tmax/dt)+1)
E = np.zeros(int(tmax/dt)+1)
ES = np.zeros(int(tmax/dt)+1)
P = np.zeros(int(tmax/dt)+1)
S[0] = S0
E[0] = E0
ES[0] = ES0
P[0] = P0

# Implement the fourth-order Runge-Kutta method
for i in range(1, len(S)):
    k1_S, k1_E, k1_ES, k1_P = derivs(S[i-1], E[i-1], ES[i-1], P[i-1])
    k2_S, k2_E, k2_ES, k2_P = derivs(S[i-1] + 0.5 * dt * k1_S,
                                       E[i-1] + 0.5 * dt * k1_E,
                                       ES[i-1] + 0.5 * dt * k1_ES,
                                       P[i-1] + 0.5 * dt * k1_P)
    k3_S, k3_E, k3_ES, k3_P = derivs(S[i-1] + 0.5 * dt * k2_S,
                                       E[i-1] + 0.5 * dt * k2_E,
                                       ES[i-1] + 0.5*dt * k2_ES,
                                       P[i-1] + 0.5 * dt * k2_P)
    k4_S, k4_E, k4_ES, k4_P = derivs(S[i-1] + dt*k3_S,
                                       E[i-1] + dt * k3_E,
                                       ES[i-1] + dt * k3_ES,
                                       P[i-1] + dt * k3_P)
    S[i] = S[i-1] + (dt / 6.0) * (k1_S + 2 * k2_S + 2 * k3_S + k4_S)
    E[i] = E[i-1] + (dt / 6.0) * (k1_E + 2 * k2_E + 2 * k3_E + k4_E)
    ES[i] = ES[i-1] + (dt / 6.0) * (k1_ES + 2 * k2_ES + 2 * k3_ES + k4_ES)
    P[i] = P[i-1] + (dt / 6.0) * (k1_P + 2 * k2_P + 2 * k3_P + k4_P)

t = [i for i in np.arange(0.0, tmax + dt, dt)]
plt.figure()
plt.plot(t, E, label='E')
plt.plot(t, S, label='S')
plt.plot(t, ES, label='ES')
plt.plot(t, P, label='P')
plt.xlabel('Time (min)')
plt.ylabel('Concentration (µM)')
plt.legend()
plt.savefig('Q22.png', dpi=300)


plt.figure()
v = [(P[i + 1] - P[i]) / dt for i in range(len(P) - 1)]
idx = np.argmax(v)
plt.scatter(S[idx], v[idx], c='r', marker='*', s=50)
plt.text(S[idx], v[idx], f'(%.3f, %.3f)' % (S[idx], v[idx]))
plt.xlabel('Concentration of S (μM)')
plt.ylabel('V (μM / min)')
plt.plot(S[1:], v)
plt.savefig('Q23.png', dpi=300)
plt.show()
