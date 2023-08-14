import numpy as np

class TournamentSelection():
    ''' Tournament selection with variable size tournament'''

    def __init__(self, tournament_prob, tournament_size, replace=True):

        assert 1.0 >= tournament_prob >= 0.0, "tournament_prob has to be in the range [0,1]"
        assert tournament_size > 0, "tournament_size has to be a positive integer"
        assert isinstance(tournament_size, int), "tournament_size has to be an integer"
        
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
        return np.random.choice(len(fitness_scores), size=1, p=fitness_scores)


