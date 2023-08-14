import random
import numpy as np
import fitness_selectors
import matplotlib.pyplot as plt
from error_check import error_check
from collections import defaultdict

class LinearGeneticProgramming():

    def __init__(self, 
                 object_fn, 
                 n_chromosomes, 
                 initial_chromosome_length,
                 n_registers,
                 C,
                 selector=fitness_selectors.TournamentSelection(0.75, 8, replace=True), 
                 p_c=0.7,
                 verbatim=False,
                 elitism=True):

        error_check(args=locals(), algorithm="LGP")

        self.object_fn = object_fn # f(x)
        self.n_variables = object_fn.__code__.co_argcount # n
        self.variable_names = object_fn.__code__.co_varnames 
        self.n_chromosomes = n_chromosomes # population size
        self.n_registers = n_registers
        self.initial_chromosome_length = initial_chromosome_length
        self.selector = selector
        self.elitism = elitism
        self.verbatim = verbatim
        self.F_best = None
        self.p_c = p_c # crossover probability
        self.C = C # constant registers
        self.operations = ["+","-"]

        # Private variables
        self.N = n_chromosomes

        # History
        self.history: dict[list] = defaultdict(list)

        self.initialize_population()

    def initialize_population(self):
        """ Initializes N chromosomes for start population, where chromsome.shape = (N,m) """

        self.population = []

        for _ in range(self.N):
            for _ in range(0,self.initial_chromosome_length,4):
                chromosome = []
                operation = random.choice(self.operations)
                destination = random.choice(range(self.n_registers))
                operand1 = random.choice(self.operands)
                operand2 = random.choice(self.operands)
                chromosome += [operation, destination, operand1, operand2]
            self.population.append(chromosome)

    def fit(self, n_epochs: int):
        """ Executes the LGP algorithm """
        
        for epoch in range(n_epochs):

            # Evaluate population
            self.evaluate_population()

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

            # Append to history
            self.update_hist()


    def decode(self):
        ''' Decodes chromosomes into variables into given range. output size (N,n) '''
        pass

        
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
    
    def update_hist(self):
        self.history["best_chromosome"].append(self.best_chromosome)
        self.history["best_variable"].append(self.best_variables)
        self.history["F_best"].append(self.F_best)




if __name__ == '__main__':

    obj_fun = lambda x,y: -((x**2+y-11)**2 + (x+y**2-7)**2)

    LGP = LinearGeneticProgramming(
        object_fn=obj_fun,
        n_chromosomes=100,
        selector=fitness_selectors.TournamentSelection(tournament_prob=0.75, tournament_size=10),
        verbatim=True,
        c_mult= 1,
    )

    LGP.fit(1000)
    print(LGP.best_variables)
    print(LGP.best_chromosome)
    print(LGP.F_best)


    N = 100
    x = np.linspace(-5,5,N)
    X,Y = np.meshgrid(x,x)
    Z = np.log((X**2+Y-11)**2 + (X+Y**2-7)**2)
    plt.contourf(X,Y,Z)
    x_best, y_best = LGP.best_variables
    plt.scatter(x_best, y_best, marker="x", s=40, c="black")
    plt.show()

