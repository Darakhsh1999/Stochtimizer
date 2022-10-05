import numpy as np
import types

class PSO():

    '''
    Particle Swarm Optimization (PSO)
    ---------------------------------
    Minimizes the given objective function (lambda function)
    
    '''
      
    def __init__(self, objective_func, x_min, x_max, n_individuals = 30, inertia = 1.4, beta= 0.999, c1= 2, c2= 2, dt= 1, alpha= 1, verbatim = 0): 
        
        # Error handling
        self.ErrorCheck(args= locals())

        # Initilize variables & constants
        self.objective_func = objective_func
        self.dimensionality = objective_func.__code__.co_argcount # n
        self.variable_names = objective_func.__code__.co_varnames 
        self.n_individuals = n_individuals # N
        self.inertia = inertia
        self.beta = beta
        self.c1 = c1
        self.c2 = c2
        self.dt = dt
        self.alpha = alpha
        self.v_max = (x_max-x_min)/(2*self.dt)
        self.verbatim = verbatim
        self.iterations = 0

        self.population = self.InitializePopulation(n_individuals, self.dimensionality, x_min, x_max) # (N, n) 
        self.velocity = self.InitializeVelocity(n_individuals, self.dimensionality, x_min, x_max, alpha, dt) # (N, n) 

        # Best score
        self.particle_best = 1e10*np.ones(n_individuals) # (n,)
        self.swarm_best = 1e10 # (,)
        self.particle_best_pos = np.zeros((self.n_individuals, self.dimensionality)) # (N, n)
        self.swarm_best_pos = np.zeros(self.dimensionality) # (3,)

        return


    def ErrorCheck(self, args):
        
        ''' Does weak checking of the input arguments '''

        print(args['objective_func'].__name__)


        if not args['objective_func'].__name__ == "<lambda>" or not isinstance(args['objective_func'], types.LambdaType):
            raise ValueError("objective_func variable must be a lambda function")
        elif not isinstance(args['n_individuals'], int) or args['n_individuals'] <= 0:
            raise ValueError("Population must be a positive integer")
        elif args['x_max'] < args['x_min']:
            raise ValueError("x_max must be larger than x_min arguments")
        elif not 0 < args['alpha'] <= 1:
            raise ValueError("Alpha parameter must be in interval (0,1]")
        elif args['dt'] <= 0:
            raise ValueError("Time step must be positive float")
        elif args['c1'] <= 0 or args['c2'] <= 0:
            raise ValueError("c1 and c2 parameters must be positive floats")
        elif args['beta'] <= 0:
            raise ValueError("Beta parameter must be positive float")
        elif args['inertia'] <= 0:
            raise ValueError("Inertia parameter must be positive float")



    def InitializePopulation(self, n_individuals, n_dimension, x_min, x_max):

        ''' Uniformly initializes swarm population in [x_min, x_max] '''

        r = np.random.rand(n_individuals, n_dimension) # (N,n)
        return  x_min + r*(x_max-x_min)


    def InitializeVelocity(self, n_individuals, n_dimension, x_min, x_max, alpha, dt):

        ''' Initialize velocities '''

        r = np.random.rand(n_individuals, n_dimension) #  (N,n)
        return  (alpha/dt)*((x_min-x_max)/2+ r*(x_max-x_min))


    def EvaluatePopulation(self):

        population_dictionary = {}

        for variable_idx, variable_key in enumerate(self.variable_names):
            population_dictionary[variable_key] = self.population[:,variable_idx]

    
        self.fitness = self.objective_func(*population_dictionary.values())


        # Update best positions
        for particle_idx, particle_score in enumerate(self.fitness):

            if particle_score < self.particle_best[particle_idx]: # particle best
                self.particle_best[particle_idx] = particle_score
                new_pos = self.population[particle_idx,:].copy()
                self.particle_best_pos[particle_idx,:] = new_pos
                
                if particle_score < self.swarm_best:  # swarm best
                    self.swarm_best = particle_score
                    self.swarm_best_pos = new_pos
                    

    def UpdateVelocity(self):
        r = np.random.rand(self.n_individuals, self.dimensionality)
        q = np.random.rand(self.n_individuals, self.dimensionality)
        self.velocity  = self.velocity*self.inertia + self.c1*q*(self.particle_best_pos-self.population)/self.dt + self.c2*r*(self.swarm_best_pos-self.population)/self.dt


    def RestrictVelocity(self):
        
        speed = np.linalg.norm(self.velocity, ord= 2, axis= 1)
        overlimit_idx = np.argwhere(speed > self.v_max)

        for idx in overlimit_idx:

            v_norm = np.linalg.norm(self.velocity[idx,:])
            self.velocity[idx,:] = (self.v_max/v_norm)*self.velocity[idx,:]


    def TrainEpoch(self):
        
        self.EvaluatePopulation()
        self.UpdateVelocity()
        self.RestrictVelocity()
        self.population = self.population + self.velocity*self.dt # update position
        self.inertia = max(0.4, self.inertia*self.beta)


    def Fit(self, epochs):

        for epoch in range(epochs):     
            self.TrainEpoch()
            self.iterations += 1
            if self.verbatim:
                print("Epoch",1+epoch, "with best fitness", self.swarm_best)



if __name__ == '__main__':

    def Succ():
        return 3

    objective_func = lambda x,y: (x**2+y-11)**2 + (x+y**2-7)**2


    PSO_optimizer = PSO(objective_func= objective_func, x_min=-5, x_max= 5, n_individuals= 30, verbatim= 0)
    PSO_optimizer.Fit(50)

    print("Training done \n --------------")
    print(PSO_optimizer.swarm_best)
    print(PSO_optimizer.swarm_best_pos)

