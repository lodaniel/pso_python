# https://github.com/ujjwalkhandelwal/pso_particle_swarm_optimization
import sys
print('Running on Python version: {}'.format(sys.version))
import numpy as np
import matplotlib.pyplot as plt

def fitness_1(X):
    x, y = X[0][0], X[1][0]
    f = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    return f
def fitness_2(X):
    x, y = X[0][0], X[1][0]
    f = (x + 2*y - 7)**2 + (2*x + y - 5)**2
    return f
def fitness_3(X):
    x, y = X[0][0], X[1][0]
    f = (1.5-x+x*y)**2 + (2.25-x+x*(y**2))**2 + (2.625-x+x*(y**3))**2
    return f
def fitness_4(X):
    x, y = X[0][0], X[1][0]
    f = 2*x*y + 2*x - x**2 - 2*(y**2)
    return f

class Particle:
    def __init__(self, n=2, vmax=1, X0=None, bound=None):
        if X0 is None:
            self.X = 2*np.random.rand(n,1) - 1
        else:
            self.X = np.array(X0, dtype='float64').reshape(-1,1)
        self.bound = bound
        self.n = n
        self.vmax = vmax
        self.V = 2*vmax*np.random.rand(n,1) - vmax
        self.clip_X()
        # inicializa 'pbest' com uma copia de 'X' 
        self.pbest = self.X.copy()
    def clip_X(self):
        if self.bound is not None:
            for i in range(self.n):
                xmin, xmax = self.bound[i]
                self.X[i,0] = np.clip(self.X[i,0], xmin, xmax)
    def update_velocity(self, w, c1, c2, gbest):
        self.clip_X()
        self.V = w*self.V # movimento anterior da particula
        self.V += c1*np.random.rand()*(self.pbest - self.X) # veloc cognitiva 
        self.V += c2*np.random.rand()*(gbest - self.X) # veloc social
        self.V = np.clip(self.V, -self.vmax, self.vmax) 
    def update_position(self):
        self.X += self.V 
        self.clip_X()

class PSO:
    def __init__(self, fitness, P=30, n=2, w=0.72984, c1=2.8, c2=2.05, 
                 Tmax=300, vmax=1, X0=None, bound=None, update_w=False, 
                 update_c1=False, update_c2=False, update_vmax=False, 
                 plot=False, min=True, verbose=False):
        self.fitness = fitness
        self.P = P 
        self.n = n 
        self.w = w
        self.c1, self.c2 = c1, c2
        self.Tmax = Tmax
        self.vmax = vmax
        self.X = X0
        self.bound = bound
        self.update_w = update_w 
        self.update_c1 = update_c1
        self.update_c2 = update_c2
        self.update_vmax = update_vmax
        self.plot = plot
        self.min = min
        self.verbose = verbose
    def optimum(self, best, particle_x):
        if self.min:
            if self.fitness(best) > self.fitness(particle_x):
                best = particle_x.copy()
        else:
            if self.fitness(best) < self.fitness(particle_x):
                best = particle_x.copy()
        return best
    def initialize(self):
        self.population = []
        for i in range(self.P):
            self.population.append(Particle(n=self.n, 
            vmax=self.vmax, X0=self.X, bound=self.bound))
            if i==0:
                self.gbest = self.population[0].X.copy()
            else:
                self.gbest = self.optimum(self.gbest, self.population[i].X)
    def update_coeff(self):
        if self.update_w:
            self.w = 0.9 - 0.5*(self.t/self.Tmax)
        if self.update_c1:
            self.c1 = 3.5 - 3*(self.t/self.Tmax)
        if self.update_c2:
            self.c2 = 0.5 + 3*(self.t/self.Tmax)
        if self.update_vmax:
            self.vmax = 1.5*np.exp(1-((self.t/self.Tmax)))
    def move(self):
        self.t = 0
        self.fitness_time, self.time = [], []
        while self.t <= self.Tmax:
            self.update_coeff()
            for particle in self.population:
                particle.update_velocity(self.w, self.c1, self.c2, self.gbest)
                particle.update_position()
                particle.pbest = self.optimum(particle.pbest, particle.X)
                self.gbest = self.optimum(self.gbest, particle.X)
            self.fitness_time.append(self.fitness(self.gbest))
            self.time.append(self.t)
            if self.verbose:
                print('Iteration:  ',self.t,'| best global fitness (cost):',
                      round(self.fitness(self.gbest),7))
            self.t += 1
    def execute(self):
        self.initialize()
        self.move()
        print('\nOPTIMUM SOLUTION\n  >', np.round(self.gbest.reshape(-1),7).tolist())
        print('\nOPTIMUM FITNESS\n  >', np.round(self.fitness(self.gbest),7))
        print()
        if self.plot:
            self.Fplot()
    def Fplot(self):
        # plota valor fitness (ou custo) versus iteracao
        plt.plot(self.time, self.fitness_time)
        plt.title('Fitness value vs Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness value')
        plt.show()

PSO(fitness=fitness_1, X0=[1,1], bound=[(-4,4),(-4,4)]).execute()
PSO(fitness=fitness_2, Tmax=50, verbose=True).execute()