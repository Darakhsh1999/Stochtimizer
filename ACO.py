import numpy as np
import matplotlib.pyplot as plt
from error_check import error_check

class AntColonyOptimization():


    def __init__(
        self,
        X: np.ndarray,
        mode: str = "AS",
        N: int = None,
        alpha: float = 1.0,
        beta: float = 2.5,
        rho: float = 0.5,
        verbatim: bool = False
        ):

        # Error handling
        error_check(args=locals(), algorithm="ACO")

        # Constants
        self.X: np.ndarray = X # (nodes, dim)
        self.n, self.dim = X.shape # n = nodes
        self.N = N if N is not None else self.n # N = n_ants
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.verbatim = verbatim

        self.visibility_matrix()
        self.nearest_neighbour_tour()

        self.tau_max = 1/(self.rho*self.D_nn)
        self.tau_min = self.tau_max*((1-0.05**(1/self.n))/((self.n/2-1)*(0.05**(1/self.n))))

        self.initialize_tau()

        # Best tour variables
        self.D_best = np.inf
        self.tour_best = []

    def visibility_matrix(self):
        ''' Initializes the visibility matrix eta and distance matrix D'''

        eta =  np.zeros((self.n,self.n)) 
        D = np.zeros((self.n,self.n)) # distance matrix, symmetric

        for i in range(self.n):
            for j in range(i):

                d_ij = np.linalg.norm(self.X[i,:]-self.X[j,:], ord=2)
                d_ij2 = np.sqrt(((self.X[i,:]-self.X[j,:])**2).sum())
                assert np.abs(d_ij-d_ij2) < 0.001, "Weird implementation" # TODO remove
                eta_ij = 1/d_ij

                D[i,j] = d_ij
                D[j,i] = d_ij
                eta[i,j] = eta_ij
                eta[j,i] = eta_ij

        self.eta = eta
        self.D = D


    def nearest_neighbour_tour(self):

        ''' Calculates the nearest neighbour tour and it's distance D_nn '''

        all_nodes = np.arange(self.n)
        start_node = np.random.choice(all_nodes) # random start node
        current_node = start_node
        tour = [current_node]
        
        for _ in range(self.n-1):

            unvisited_nodes = np.setdiff1d(all_nodes, tour) # index of unvisted nodes
            unvisted_min_idx = np.argmin(self.D[unvisited_nodes, current_node]) # high eta = low distance

            next_node = unvisited_nodes[unvisted_min_idx]
            tour.append(next_node)
            current_node = next_node
        
        tour.append(start_node) # close the path
        
        D_nn = self.get_path_length(tour)

        self.D_nn = D_nn
        self.tour_nn = tour

    def initialize_tau(self):
        ''' Initializes the pheromone matrix tau'''

        ones = np.ones((self.n,self.n))
        if self.mode == "AS":
            self.tau = (self.N/self.D_nn)*ones
        else: # MMAS
            self.tau = self.tau_max*ones

    def fit(self, n_epochs):
        ''' Runs the main ACO algorithm using the given mode '''

        for epoch in range(n_epochs):

            if self.mode == "AS": # Ant system

                delta_tau = np.zeros((self.n,self.n))

                for ant_k in range(self.N): # Generate paths
                    
                    tour_k = self.generate_tour()
                    D_k = self.get_path_length(tour_k)

                    if D_k < self.D_best: # Found new best path
                        self.D_best = D_k
                        self.tour_best = tour_k
                        if self.verbatim:
                            print(f"New best path found in epoch {epoch} from ant {ant_k} with path length D_k = {D_k:.3f}")


                    delta_tau += self.delta_tau_k(tour_k, D_k)

                # Update pheromone matrix tau
                self.tau = (1-self.rho)*self.tau + delta_tau 

            else: # Max-min ant system
                
                D_mmas = np.inf

                for ant_k in range(self.N): 
                    
                    tour_k = self.generate_tour()
                    D_k = self.get_path_length(tour_k)

                    if D_k < D_mmas: # best tour this iteration

                        tour_mmas = tour_k
                        D_mmas = D_k

                        if D_k < self.D_best: # best tour so far

                            self.tour_best = tour_k
                            self.D_best = D_k
                            if self.verbatim:
                                print(f"New best path found in epoch {epoch} from ant {ant_k} with path length D_k = {D_k:.3f}")

                # Use best ant for each iteration for pheromone update
                delta_tau = np.zeros((self.n,self.n))
                delta_tau[tour_mmas[1:], tour_mmas[:-1]] = 1/D_mmas
                self.tau = (1-self.rho)*self.tau + delta_tau # Update pheromones
                
                # Impose limits
                self.tau_limits()

    def generate_tour(self):
        ''' Generates a TSP tour from a randomly selected started node '''

        all_nodes = np.arange(self.n)
        start_node = np.random.choice(all_nodes) # random start node
        current_node = start_node
        tour = [current_node]
        
        for _ in range(self.n-1):

            unvisited_nodes = np.setdiff1d(all_nodes, tour) # index of unvisted nodes

            next_node = self.next_node(current_node, unvisited_nodes)
            tour.append(next_node)
            current_node = next_node

        tour.append(start_node)

        return tour

    def next_node(self, current_node, unvisited_nodes):
        ''' Performs tournament selection among the unvisted nodes '''

        tau_eta = (self.eta[unvisited_nodes, current_node]**self.alpha)*(self.tau[unvisited_nodes, current_node]**self.beta)
        normalization_factor = tau_eta.sum()
        transition_probability = tau_eta / normalization_factor

        node_idx = np.random.choice(unvisited_nodes, size=1, p=transition_probability)[0]
        return node_idx

    def get_path_length(self, tour):
        ''' Calculates the path length of a given tour '''
        return self.D[tour[1:],tour[:-1]].sum()

    def delta_tau_k(self, tour_k, D_k):
        ''' Delta tau matrix for a tour'''
        delta_tau_k = np.zeros((self.n,self.n))
        delta_tau_k[tour_k[1:], tour_k[:-1]] = 1/D_k
        return delta_tau_k
    
    def tau_limits(self):
        ''' Updates tau limits for MMAS '''
        self.tau[self.tau < self.tau_min] = float(self.tau_min)
        self.tau[self.tau > self.tau_max] = float(self.tau_max)
        self.tau_max = 1/(self.rho*self.D_best)
        self.tau_min = self.tau_max*((1-0.05**(1/self.n))/((self.n/2-1)*(0.05**(1/self.n))))

    def plot_tsp(self):

        if self.dim == 2:
            plt.scatter(X[self.tour_best[0],0], X[self.tour_best[0],1], marker="x", c="k", s=200) # start
            plt.scatter(X[:,0],X[:,1]) # node
            for node_idx, n1 in enumerate(self.tour_best): # Add arrows
                if node_idx == self.n: break
                n2 = self.tour_best[node_idx+1]
                x_arrow, y_arrow = self.X[n1,:]
                dx_arrow, dy_arrow = (self.X[n2,:]-self.X[n1,:])
                plt.arrow(x_arrow,y_arrow, dx_arrow, dy_arrow, length_includes_head=True, width=0.003)
            plt.grid()
            plt.xlim([0,1.0])
            plt.ylim([0,1.0])
            plt.title(f"Shortest path $D_k$ = {self.D_best:.3f}")
            plt.legend(["Start","Nodes","Path"], loc="upper right")

            plt.show()
        else:
            raise Exception("Plotting is only supported for 2 dimensional data")


if __name__ == '__main__':
    X = np.random.rand(30,2)
    ACO = AntColonyOptimization(X=X, N=40, mode="AS", verbatim=True)
    ACO.fit(100)
    ACO.plot_tsp()

