import numpy as np
import Selectors


class GA():

    def __init__(self, 
                 object_func, 
                 variable_length, # k
                 n_chromosomes, # n
                 selector=  Selectors.TournamentSelection(0.75, 8, replace= True), 
                 encoding= 'binary', 
                 range= 10, # d
                 c_mult= 1, 
                 p_c= 0.7,
                 verbatim = 0):

        self.ErrorCheck(args = locals())

        self.object_func = object_func
        self.n_variables = object_func.__code__.co_argcount
        self.variable_names = object_func.__code__.co_varnames 
        self.chromosome_length = self.n_variables*variable_length
        self.variable_length = variable_length # number of genes each variable is decoded into
        self.n_chromosomes = n_chromosomes # population size
        self.encoding = encoding
        self.selector = selector
        if selector.__class__ == 'Selectors.RouletteWheelSelection': # used for tournament selection
            self.selector.SetSize(n_chromosomes) 
        self.range = range
        self.p_c = p_c
        self.p_mut = c_mult/(self.chromosome_length)
        self.verbatim = verbatim

        # Private variables
        self._q = self.n_variables
        self._k = variable_length
        self._m = self.chromosome_length 
        self._n = n_chromosomes
        self._d = range

        self.InitializePopulation()


    def ErrorCheck(self, args):

        ''' Validates input data '''
            # if encoding is real -> check var_len =1 
        pass # check so encoding isnt real and variable_length =! 1 


    def InitializePopulation(self):
        if self.encoding == 'real':
            self.population =  np.random.rand(self._n, self._m)
        elif self.encoding == 'binary':
            self.population = np.random.randint(0, 2, size= (self._n, self._m))
        else:
            raise ValueError("Invalid encoding used")


    def RealDecoding(self):
        
        ''' Decodes chromosomes into variables
            into given range. output size (n,q) '''

        return -self._d + 2*self._d*self.population


    def BinaryDecoding(self):

        ''' Decodes chromosomes into variables
            into given range. output size (n,q) '''

        variable = np.zeros(shape= (self._n, self._q))
        bin_conversion = np.reciprocal(np.power(2,np.arange(1, self._k+1), dtype= float)) 

        for chro_idx in range(self._n):
            for var_idx in range(self._q):
                variable[chro_idx, var_idx] = (bin_conversion*self.population[chro_idx, var_idx*self._k:(var_idx+1)*self._k]).sum()

        return -self.range + (2*self.range)/(1-2**(-self._k))*variable


    def NewGeneration(self):
        
        new_population = np.zeros(shape= (self._n, self._m))
        
        ## Elitism
        if self._n % 2 == 0: 
            new_population[0:2,:] = self.best_chromosome
            offset = 2
        else:
            new_population[0,:] = self.best_chromosome
            offset = 1

        # Rest of new generation 
        for i in range (offset,self._n-1,2):
            
            # Chromosome pair
            c1, c2 = self.PickPair()

            # Crossover
            if np.random.rand() < self.p_c:
                c1, c2 = self.Crossover(c1,c2)
            
            # Mutate
            if np.random.rand() < self.p_mut:
                c1 = self.Mutate(c1)
            if np.random.rand() < self.p_mut:
                    c2 = self.Mutate(c2)

            new_population[i,:] = c1
            new_population[i+1,:] = c2 

        return new_population


    def PickPair(self):

        ''' Picks chromosome pair using selector '''

        c1_idx, c2_idx = self.selector.Select(self.fitness), self.selector.Select(self.fitness)
        c1, c2 = self.population[c1_idx,:], self.population[c2_idx,:]
        
        return c1, c2


    def Crossover(self, chromosome1, chromosome2):
    
        ''' Performs crossover on 2 chromosomes
            with single crossover point '''

        cutoff = np.random.randint(0, len(chromosome1)-1)

        new_chromosome1 = np.concatenate((chromosome1[:cutoff+1], chromosome2[cutoff+1:]))
        new_chromosome2 = np.concatenate((chromosome2[:cutoff+1], chromosome1[cutoff+1:]))

        return (new_chromosome1, new_chromosome2)
        

    def Mutate(self, chromosome):

        m_idx = np.random.rand(self._m) < self.p_mut

        if self.encoding == 'real':
            chromosome[m_idx] = np.random.rand(len(m_idx))
        else:
            chromosome[m_idx] = 1 - chromosome[m_idx]

        return chromosome


    def Fit(self, epochs):
        
        for epoch in range(epochs):

            # Decoding
            if self.encoding == 'real':
                self.variables = self.RealDecoding()
            else:
                self.variables = self.BinaryDecoding()

            # Evaluate population
            variable_dictionary = {}

            for variable_idx, variable_key in enumerate(self.variable_names):
                variable_dictionary[variable_key] = self.variables[:,variable_idx]

            self.fitness = self.object_func(*variable_dictionary.values())

            # Update best individual
            self.F_best = np.amax(self.fitness)
            self.i_best = np.argmax(self.fitness)
            self.best_chromosome = self.population[self.i_best,:].copy() 

            if self.verbatim:
                print(f"Best fitness in epoch {epoch} is F_max = {self.F_best}")

            # New population
            self.population = self.NewGeneration()


if __name__ == '__main__':

    obj_fun = lambda x,y: -((1.5-x+x*y)**2 + (2.25-x+x*y**2)**2 + (2.625-x+x*y**3)**2)

    GA_obj = GA(object_func= obj_fun, variable_length= 16, n_chromosomes= 100, selector= Selectors.RouletteWheelSelection(), verbatim= 1)
    GA_obj.Fit(50)
    print(GA_obj.variables[GA_obj.i_best, :])

