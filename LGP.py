import random
import sympy
import numpy as np
import fitness_selectors
import matplotlib.pyplot as plt
from error_check import error_check
from collections import defaultdict
from operations import Operations

class LinearGeneticProgramming():

    def __init__(self, 
        object_fn, 
        operation_names,
        N, 
        initial_chromosome_length,
        n_registers,
        C,
        selector=fitness_selectors.TournamentSelection(0.75, 8, replace=True), 
        p_c=0.7,
        max_length=160,
        penalty_length=80,
        verbatim=False,
        elitism=True
        ):

        #error_check(args=locals(), algorithm="LGP")

        self.object_fn = object_fn # f(x)
        self.operation_names = operation_names
        self.operations = Operations(self.operation_names).get_operations() # idx to operation
        self.n_operations = len(self.operations)
        self.N = N
        self.initial_chromosome_length = initial_chromosome_length
        self.n_registers = n_registers
        self.C = C # constant registers
        self.n_operands = self.n_registers + len(C)
        self.selector = selector
        self.p_c = p_c # crossover probability
        self.max_length = max_length
        self.penalty_length = penalty_length
        self.verbatim = verbatim
        self.elitism = elitism

        self.F_best = np.inf

        # History
        self.history: dict[list] = defaultdict(list)

        self.initialize_population()

    def initialize_population(self):
        """ Initializes N chromosomes for start population, where chromsome.shape = (N,m) """

        self.population = []

        for _ in range(self.N):
            chromosome = []
            for _ in range(0,self.initial_chromosome_length,4):
                operation = random.choice(range(self.n_operations))
                destination = random.choice(range(self.n_registers))
                operand1 = random.choice(range(self.n_operands))
                operand2 = random.choice(range(self.n_operands))
                chromosome += [operation, destination, operand1, operand2]
            self.population.append(chromosome)

    def fit(self, n_epochs: int, x, y):
        """ Executes the LGP algorithm """

        # Input and target
        self.x  = x
        self.y = y
        
        for epoch in range(n_epochs):

            # Evaluate population
            self.evaluate_population()

            # Update best individual
            F_epoch_best = np.min(self.fitness_scores)
            if (F_epoch_best < self.F_best): # New best found
                self.F_best = F_epoch_best 
                self.i_best = np.argmin(self.fitness_scores)
                self.best_chromosome = self.population[self.i_best].copy() 

            if self.verbatim:
                print(f"Best fitness in epoch {epoch:3} is F_max = {self.F_best:.4f}")

            # Append to history
            self.history["best_chromosome"].append(self.best_chromosome)
            self.history["F_best"].append(self.F_best)
            
            if epoch == n_epochs:
                break

            # Sample next population generation
            self.next_generation()

        
    def evaluate_population(self):
        
        fitness_scores = np.zeros(self.N)

        for i in range(self.N):

            chromosome = self.population[i]
            y_pred = np.zeros(len(self.y))

            for idx, x in enumerate(self.x): # loop through data
                y_pred[idx] = self.operate(x, chromosome)

            fitness_scores[i] = self.object_fn(y_pred, self.y)
            

        penalty = np.median(fitness_scores)*np.array([max(0,len(x)-self.penalty_length) for x in self.population])
        self.fitness_scores = fitness_scores + penalty


    def operate(self, x, chromosome):
        """ Performs operations according to chromosome """

        A = [float(x)] + (self.n_registers-1)*[0.0] + self.C # registers
        skip_next = False
        
        for i in range(0,len(chromosome),4):

            if skip_next: # conditional branch
                skip_next = False
                continue

            operation = chromosome[i]
            destination = chromosome[i+1]
            operand1 = A[chromosome[i+2]]
            operand2 = A[chromosome[i+3]]

            result = self.operations[operation](operand1,operand2)
            if isinstance(result, float):
                A[destination] = result
            elif isinstance(result, bool):
                skip_next = result
            else:
                raise TypeError(f"Got unexpected type {type(result)} from operation")
        
        return A[0] # r_0 = output

    def decode_function(self, chromosome=None):

        x = sympy.symbols("x")
        A = [x] + (self.n_registers-1)*[0.0] + self.C # registers

        if chromosome is None:
            chromosome = self.best_chromosome
        
        for i in range(0,len(chromosome),4):

            operation = chromosome[i]
            destination = chromosome[i+1]
            operand1 = A[chromosome[i+2]]
            operand2 = A[chromosome[i+3]]

            A[destination] = self.operations[operation](operand1, operand2, sym=True)
        
        return A[0] # r_0 = output


    def next_generation(self):
        
        next_population = []
        
        ## Elitism
        if self.elitism:
            if self.N % 2 == 0: 
                next_population.append(self.best_chromosome.copy())
                next_population.append(self.best_chromosome.copy())
                ptr = 2
            else:
                next_population.append(self.best_chromosome.copy())
                ptr = 1
        else:
            ptr = 0

        # Rest of new generation 
        for _ in range (ptr,self.N-1,2):
            
            # Chromosome pair
            c1, c2 = self.select_pair()

            # Crossover
            if np.random.rand() < self.p_c:
                c1, c2 = self.crossover(c1,c2)
            
            # Mutate
            c1 = self.mutate(c1)
            c2 = self.mutate(c2)

            # Write
            next_population.append(c1)
            next_population.append(c2)
        
        
        self.population = next_population

    def select_pair(self):
        ''' Picks chromosome pair using selector '''
        c1_idx = self.selector.select(self.fitness_scores)
        c2_idx = self.selector.select(self.fitness_scores)
        return self.population[c1_idx].copy(), self.population[c2_idx].copy()

    def crossover(self, chromosome1, chromosome2):
        ''' Performs crossover on 2 chromosomes with single crossover point '''

        n_cutoff_points1 = len(chromosome1)//4 
        n_cutoff_points2 = len(chromosome2)//4 
        cutoff11 = random.randint(0, n_cutoff_points1-1) 
        cutoff12 = random.randint(1+cutoff11, n_cutoff_points1 if cutoff11 != 0 else n_cutoff_points1-1) 
        cutoff21 = random.randint(0, n_cutoff_points2-1) 
        cutoff22 = random.randint(1+cutoff21, n_cutoff_points2 if cutoff21 != 0 else n_cutoff_points2-1) 
        cutoff11 *= 4
        cutoff12 *= 4
        cutoff21 *= 4
        cutoff22 *= 4

        new_chromosome1 = chromosome1[:cutoff11] + chromosome2[cutoff21:cutoff22] + chromosome1[cutoff12:]
        new_chromosome2 = chromosome2[:cutoff21] + chromosome1[cutoff11:cutoff12] + chromosome2[cutoff22:]

        return (new_chromosome1[:self.max_length], new_chromosome2[:self.max_length])

    def mutate(self, chromosome):

        mutate_idx = np.random.rand(len(chromosome)) < 1.0/len(chromosome)

        if len(mutate_idx) == 0: return chromosome # no mutation

        for idx in np.argwhere(mutate_idx).flatten():

            if idx % 4 == 0: # operation
                mutated_gene = random.choice(range(self.n_operations))
            elif idx % 4 == 1: # destination
                mutated_gene = random.choice(range(self.n_registers))
            else: # operand
                mutated_gene = random.choice(range(self.n_operands))

            chromosome[idx] = mutated_gene

        return chromosome

    def predict_best_chromosome(self):

        y_pred = np.zeros(len(self.y))

        for idx, x in enumerate(self.x): # loop through data
            y_pred[idx] = self.operate(x, self.best_chromosome)

        return y_pred
    


if __name__ == '__main__':

    obj_fun = lambda x,y: ((x-y)**2).mean()
    operation_names = ["addition","subtraction","multiplication","square"]

    n_points = 100
    x_data = np.linspace(0,30,n_points)
    y_data = 2*x_data**2 + 3

    LGP = LinearGeneticProgramming(
        object_fn=obj_fun,
        operation_names=operation_names,
        N=100,
        initial_chromosome_length=16,
        n_registers=3,
        C=[1.0,2.0,3.0,-1.0],
        selector=fitness_selectors.TournamentSelection(tournament_prob=0.75, tournament_size=10),
        p_c=0.8,
        max_length=40,
        penalty_length=32,
        verbatim=True,
        elitism=True
    )

    LGP.fit(600, x_data, y_data)
    print(f"L_c = {len(LGP.best_chromosome)}, chromosome: {LGP.best_chromosome}")
    y_predict = LGP.predict_best_chromosome()
    print(f"Final loss = {LGP.object_fn(y_data,y_predict):.4f}")
    expression = LGP.decode_function()
    print(sympy.expand(expression))

    plt.plot(x_data,y_data,"bo-")
    plt.plot(x_data,y_predict,"ro--")
    plt.legend(["Data","LGP"])
    plt.title(f"Predicted function: {expression}")
    plt.grid()
    plt.show()
