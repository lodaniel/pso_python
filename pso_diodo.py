# Exemplo no site da biblioteca Pyswarms:
# https://pyswarms.readthedocs.io/en/latest/examples/usecases/electric_circuit_problem.html
import sys
import numpy as np
import matplotlib.pyplot as plt
import pyswarms as ps
def cost_func(I):
    # Parametros fixos
    U = 10
    R = 100
    I_s = 9.4e-12
    v_t = 25.85e-3
    c = abs(U - v_t * np.log(abs(I[:, 0] / I_s)) - R * I[:, 0])
    return c
# Setando hiperparametros
options = {'c1': 0.5, 'c2': 0.3, 'w':0.3}
# chamando PSO
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=1, options=options)
# Fazendo a otimizacao
cost, pos = optimizer.optimize(cost_func, iters=30)
print(pos[0])
print(cost)
x = np.linspace(0.001, 0.1, 100).reshape(100, 1)
y = cost_func(x)
plt.plot(x, y)
plt.xlabel('Corrente I [A]')
plt.ylabel('Custo');
plt.show()
# Importando o solucionador nao-linear
from scipy.optimize import fsolve
c = lambda I: abs(10 - 25.85e-3 * np.log(abs(I / 9.4e-12)) - 100 * I)
initial_guess = 0.09
corr_I = fsolve(func=c, x0=initial_guess)
print(corr_I[0])
