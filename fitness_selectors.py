import numpy as np

class TournamentSelection():
    ''' Tournament selection with variable size tournament'''

    def __init__(self, tournament_prob, tournament_size, replace=True):
        
        self.p_tour = tournament_prob
        self.size = tournament_size
        self.replace = replace

    def select(self, fitness_scores):

        ''' Performs 1 selection '''

        tournament = np.random.choice(np.arange(len(fitness_scores)), self.size, replace=self.replace) # index of tournament individuals 
         

        while True:

            tournament_fitness = fitness_scores[tournament]
            r = np.random.rand()
            best_tournament_idx = np.argmax(tournament_fitness) # index of best individual w.r.t tournament array
            best_individual_idx = tournament[best_tournament_idx] # idx of best indiviual in tournament

            if r < self.p_tour or len(tournament) == 1: # Select best or last individual
                return best_individual_idx
            else:
                tournament = np.delete(tournament, best_tournament_idx)


class RouletteWheelSelection():

    ''' Fitness proportionate roulette wheel selection '''

    def select(self, fitness_scores):

        ''' Performs 1 selection '''
        
        r = np.random.rand()
        phi_i = 0.0
        F_tot = fitness_scores.sum()
        
        for i, F_i in enumerate(fitness_scores):
            phi_i += float(F_i/F_tot)
            if phi_i > r:
                return i 


