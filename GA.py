import numpy as np
import fitness_selectors
import matplotlib.pyplot as plt
from error_check import error_check


class GeneticAlgorithm():

    def __init__(self, 
                 object_fn, 
                 variable_length, 
                 n_chromosomes, 
                 selector=fitness_selectors.TournamentSelection(0.75, 8, replace=True), 
                 encoding='binary', 
                 range=10, 
                 c_mult=1, 
                 p_c=0.7,
                 verbatim=False,
                 elitism=True):

        error_check(args=locals(), algorithm="GA")

        self.object_fn = object_fn # f(x)
        self.n_variables = object_fn.__code__.co_argcount # n
        self.variable_names = object_fn.__code__.co_varnames 
        self.chromosome_length = self.n_variables*variable_length # m
        self.variable_length = variable_length # k
        self.n_chromosomes = n_chromosomes # population size
        self.encoding = encoding
        self.selector = selector
        self.range = range # d
        self.elitism = elitism
        self.verbatim = verbatim
        self.F_best = None

        # Private variables
        self.n = self.n_variables
        self.k = variable_length
        self.m = self.chromosome_length if self.encoding == "binary" else self.n_variables
        self.N = n_chromosomes
        self.d = range

        self.p_mut = c_mult/(self.m) # mutation probability
        self.p_c = p_c # crossover probability

        self.initialize_population()

    def initialize_population(self):
        """ Initializes N chromosomes for start population, where chromsome.shape = (N,m) """

        if self.encoding == 'real':
            self.population =  np.random.rand(self.N, self.m)
        elif self.encoding == 'binary':
            self.population = np.random.randint(0, 2, size= (self.N, self.m))
        else:
            raise ValueError(f"Invalid encoding used: {self.encoding}")

    def fit(self, n_epochs: int):
        """ Executes the GA algorithm by maximizing objective function """
        
        for epoch in range(n_epochs):

            # Decoding
            self.variables = self.decode()

            # Evaluate population
            self.fitness_scores = self.object_fn(*[x_i for x_i in self.variables.T])

            # Update best individual
            F_epoch_best = np.max(self.fitness_scores)
            if (self.F_best is None) or (F_epoch_best > self.F_best): # New best found
                self.F_best = F_epoch_best 
                self.i_best = np.argmax(self.fitness_scores)
                self.best_chromosome = self.population[self.i_best,:].copy() 
                self.best_variables = self.variables[self.i_best,:].copy()

            if self.verbatim:
                print(f"Best fitness in epoch {epoch:3} is F_max = {self.F_best:.4f}")
            
            if epoch == n_epochs:
                break

            # Sample next population generation
            self.next_generation()


    def decode(self):
        ''' Decodes chromosomes into variables into given range. output size (N,n) '''

        if self.encoding == "real":
            return -self.d + 2*self.d*self.population
        else:
            variable = np.zeros((self.N, self.n))
            bin_conversion = np.reciprocal(np.power(2,np.arange(1, self.k+1), dtype= float)) 

            for chro_idx in range(self.N):
                for var_idx in range(self.n):
                    variable[chro_idx, var_idx] = (bin_conversion*self.population[chro_idx, var_idx*self.k:(var_idx+1)*self.k]).sum()

            return -self.range + (2*self.range)/(1-2**(-self.k))*variable
        
    def next_generation(self):
        
        next_population = np.zeros((self.N, self.m))
        
        ## Elitism
        if self.elitism:
            if self.N % 2 == 0: 
                next_population[0:2,:] = self.best_chromosome.copy()
                ptr = 2
            else:
                next_population[0,:] = self.best_chromosome.copy()
                ptr = 1
        else:
            ptr = 0

        # Rest of new generation 
        for i in range (ptr,self.N-1,2):
            
            # Chromosome pair
            c1, c2 = self.select_pair()

            # Crossover
            if np.random.rand() < self.p_c:
                c1, c2 = self.crossover(c1,c2)
            
            # Mutate
            if np.random.rand() < self.p_mut:
                c1 = self.mutate(c1)
            if np.random.rand() < self.p_mut:
                c2 = self.mutate(c2)

            # Write
            next_population[i,:] = c1
            next_population[i+1,:] = c2 
        
        if self.N % 2 == 1: # Just copy the 2nd last to last
            next_population[-1,:] = c2
        
        self.population = next_population


    def select_pair(self):
        ''' Picks chromosome pair using selector '''
        c1_idx = self.selector.select(self.fitness_scores)
        c2_idx = self.selector.select(self.fitness_scores)
        return self.population[c1_idx,:].copy(), self.population[c2_idx,:].copy()
        


    def crossover(self, chromosome1, chromosome2):
        ''' Performs crossover on 2 chromosomes with single crossover point '''

        cutoff = np.random.randint(0, len(chromosome1)-1)
        new_chromosome1 = np.concatenate((chromosome1[:cutoff+1], chromosome2[cutoff+1:]))
        new_chromosome2 = np.concatenate((chromosome2[:cutoff+1], chromosome1[cutoff+1:]))

        return (new_chromosome1, new_chromosome2)
        

    def mutate(self, chromosome):

        mutate_idx = np.random.rand(self.m) < self.p_mut

        if self.encoding == 'real':
            for m_idx in np.argwhere(mutate_idx): # Creep mutation
                old_value = chromosome[m_idx] 
                new_value =  max(min(np.random.normal(loc=old_value, scale=0.1),1), 0)
                chromosome[m_idx] = new_value
        else: # Bit flip mutation
            chromosome[mutate_idx] = 1 - chromosome[mutate_idx]

        return chromosome




if __name__ == '__main__':

    obj_fun = lambda x,y: -((x**2+y-11)**2 + (x+y**2-7)**2)

    GA = GeneticAlgorithm(
        object_fn=obj_fun,
        variable_length=16,
        n_chromosomes=100,
        range=5,
        selector=fitness_selectors.TournamentSelection(tournament_prob=0.75, tournament_size=10),
        verbatim=True,
        c_mult= 1,
        encoding="binary"
    )

    GA.fit(1000)
    print(GA.best_variables)
    print(GA.best_chromosome)
    print(GA.F_best)


    N = 100
    x = np.linspace(-5,5,N)
    X,Y = np.meshgrid(x,x)
    Z = np.log((X**2+Y-11)**2 + (X+Y**2-7)**2)
    plt.contourf(X,Y,Z)
    x_best, y_best = GA.best_variables
    plt.scatter(x_best, y_best, marker="x", s=40, c="black")
    plt.show()

