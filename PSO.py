import numpy as np
import matplotlib.pyplot as plt
from error_check import error_check

class ParticleSwarmOptimization():

    '''
    Particle Swarm Optimization (PSO)
    ---------------------------------
    Minimizes the given objective function (lambda function)
    
    '''
      
    def __init__(
        self,
        object_fn,
        x_min,
        x_max,
        N=30,
        W=1.4,
        W_min=0.4,
        beta=0.999,
        c1=2,
        c2=2,
        dt=1,
        alpha=1,
        verbatim=False
    ): 
        
        # Error handling
        error_check(args=locals(), algorithm="PSO")

        # Initilize variables & constants
        self.object_fn = object_fn
        self.n = object_fn.__code__.co_argcount 
        self.N = N 
        self.W = W
        self.W_min = W_min
        self.beta = beta
        self.c1 = c1
        self.c2 = c2
        self.dt = dt
        self.x_max = x_max
        self.x_min = x_min
        self.alpha = alpha
        self.v_max = (x_max-x_min)/(self.dt)
        self.verbatim = verbatim

        self.initialize_population() # (N, n) 
        self.initialize_velocity() # (N, n) 

        # Best score
        self.particle_best = np.inf*np.ones(self.N) # (n,)
        self.particle_best_pos = np.zeros((self.N, self.n)) # (N, n)
        self.swarm_best = np.inf # (,)
        self.swarm_best_pos = np.zeros(self.n) # (3,)


    def initialize_population(self):
        ''' Uniformly initializes swarm population in [x_min, x_max] '''
        r = np.random.rand(self.N, self.n) # (N,n)
        self.population = self.x_min + r*(self.x_max-self.x_min)

    def initialize_velocity(self):
        ''' Initialize velocities '''
        r = np.random.rand(self.N, self.n) #  (N,n)
        self.velocity = (self.alpha/self.dt)*((self.x_min-self.x_max)/2+ r*(self.x_max-self.x_min))

    def evaluate_population(self):

        # Evaluate population
        self.fitness_scores = self.object_fn(*[x_i for x_i in self.population.T])

        # Update best positions
        for p_idx, p_score in enumerate(self.fitness_scores):

            if p_score < self.particle_best[p_idx]: # Particle best
                self.particle_best[p_idx] = p_score
                new_pos = self.population[p_idx,:].copy()
                self.particle_best_pos[p_idx,:] = new_pos
                
                if p_score < self.swarm_best:  # Swarm best
                    self.swarm_best = p_score
                    self.swarm_best_pos = new_pos
                    

    def update_velocity(self):
        r = np.random.rand(self.N, self.n)
        q = np.random.rand(self.N, self.n)
        C1 = (self.c1*q)/self.dt
        C2 = (self.c2*r)/self.dt
        self.velocity  = self.velocity*self.W + C1*(self.particle_best_pos-self.population) + C2*(self.swarm_best_pos-self.population)

    def restrict_velocity(self):
        speed = np.linalg.norm(self.velocity, ord=2, axis=1)
        overlimit_idx = np.argwhere(speed > self.v_max)
        for idx in overlimit_idx:
            self.velocity[idx,:] *= (self.v_max/speed[idx])

    def train_epoch(self):
        
        self.evaluate_population()
        self.update_velocity()
        self.restrict_velocity()
        self.population = self.population + self.velocity*self.dt # update position
        self.W = max(self.W_min, self.W*self.beta)


    def fit(self, n_epochs: int):

        for epoch in range(n_epochs):     
            self.train_epoch()
            if self.verbatim:
                print(f"Epoch {epoch} with best fitness F = {self.swarm_best:.3f}")
        print("PSO algorithm finished!")



if __name__ == '__main__':

    object_fn = lambda x,y: (x**2+y-11)**2 + (x+y**2-7)**2

    PSO = ParticleSwarmOptimization(object_fn=object_fn, x_min=-5, x_max=5, N=30, verbatim=True)
    PSO.fit(30)

    print(PSO.swarm_best)
    print(PSO.swarm_best_pos)

    N = 100
    x = np.linspace(-5,5,N)
    X,Y = np.meshgrid(x,x)
    Z = np.log((X**2+Y-11)**2 + (X+Y**2-7)**2)
    plt.contourf(X,Y,Z)
    x_best, y_best = PSO.swarm_best_pos
    plt.scatter(x_best, y_best, marker="x", s=40, c="black")
    plt.show()

