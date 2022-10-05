import numpy as np
import matplotlib.pyplot as plt

class ACO():


    def __init__(self, X: np.ndarray, mode:str= "AS", alpha:float= 1.0, beta:float= 2.5, rho:float= 0.5, verbatim:bool= False):

        # Error handling
        self.ErrorCheck(args= locals())

        # Constants
        self.X = X # (nodes, dim)
        self.n, self.dim = X.shape
        self.n_ants = self.n
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.verbatim = verbatim

        self.eta, self.D = self.VisibilityMatrix()
        self.D_nn, self.nn_tour = self.NearestNeighbourTour()

        self.tau_max = 1/(self.rho*self.D_nn)
        self.tau_min = self.tau_max*((1-0.05**(1/self.n))/((self.n/2)*(0.05**(1/self.n))))

        self.tau = self.InitializeTau()

        # Best tour variables
        self.D_best = np.Inf
        self.tour_best = []


    def ErrorCheck(self, args):

        X, alpha, beta, rho = args["X"], args["alpha"], args["beta"], args["rho"]
    

        def RE(var):
            err_msg = var+" must be a positive float"
            raise ValueError(err_msg)

        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array")
        elif not isinstance(alpha, float) or alpha <= 0:
            RE("alpha")
        elif not isinstance(beta, float) or beta <= 0:
            RE("beta")
        elif not isinstance(rho, float) or rho <= 0:
            RE("rho")


    def VisibilityMatrix(self):
        
        ''' Initializes the visibility matrix and distance matrix D'''

        eta =  np.zeros((self.n,self.n)) 
        D = np.zeros((self.n,self.n)) # distance matrix, symmetric

        for i in range(self.n):
            for j in range(i):

                d_ij = np.linalg.norm(self.X[i,:]-self.X[j,:], ord=2)
                eta_ij = 1/d_ij

                D[i,j] = d_ij
                D[j,i] = d_ij
                eta[i,j] = eta_ij
                eta[j,i] = eta_ij

        return eta, D


    def NearestNeighbourTour(self):

        ''' Calculates the nearest neighbour tour and it's distance D_nn '''

        all_nodes = np.arange(self.n)
        start_node = np.random.choice(all_nodes) # random start node
        current_node = start_node
        tour = [current_node]
        
        for _ in range(self.n-1):

            unvisited_nodes = np.setdiff1d(all_nodes, tour) # index of unvisted nodes
            unvisted_min_idx = np.argmin(self.D[current_node, unvisited_nodes]) # high eta = low distance

            next_node = unvisited_nodes[unvisted_min_idx]
            tour.append(next_node)
            current_node = next_node
        
        tour.append(start_node) # close the path
        
        D_nn = self.GetPathLength(tour)

        return D_nn, tour


    def InitializeTau(self):
        
        ''' Initializes the pheromone matrix tau'''

        ones = np.ones((self.n,self.n))
        
        if self.mode == "AS":
            return (self.n_ants/self.D_nn)*ones
        elif self.mode == "MMAS":
            return self.tau_max*ones
        else:
            raise ValueError("Unknown mode. Supported modes are; 'AS' and 'MMAS'.")


    def Fit(self, epochs):
        
        ''' Runs the main ACO algorithm using
            the given mode '''

        for epoch in range(epochs):

            if self.mode == "AS":

                delta_tau = np.zeros((self.n,self.n))

                for ant_k in range(self.n_ants): # Generate paths
                    
                    tour_k = self.GenerateTour()
                    D_k = self.GetPathLength(tour_k)

                    if D_k < self.D_best:

                        self.D_best = D_k
                        self.tour_best = tour_k
                        if self.verbatim:
                            print(f"New best path found in epoch {epoch} from ant {ant_k} with path length D_k = {D_k:.3f}")


                    delta_tau += self.DeltaTauK(tour_k, D_k)

                self.tau = (1-self.rho)*self.tau + delta_tau # pheromone evaporation

            else: # Max-min ant system
                
                D_mmas = np.inf

                for ant_k in range(self.n_ants): 
                    
                    tour_k = self.GenerateTour()
                    D_k = self.GetPathLength(tour_k)

                    if D_k < D_mmas: # best tour this iteration

                        D_mmas = D_k
                        tour_mmas = tour_k

                        if D_k < self.D_best: # best tour so far

                            self.D_best = D_k
                            self.tour_best = tour_k
                            if self.verbatim:
                                print(f"New best path found in epoch {epoch} from ant {ant_k} with path length D_k = {D_k:.3f}")

                # Use best ant for each iteration for pheromone update
                delta_tau = np.zeros((self.n,self.n))
                delta_tau[tour_mmas[1:], tour_mmas[:-1]] = 1/D_mmas
                self.tau = (1-self.rho)*self.tau + delta_tau # Update pheromones
                
                # Impose limits
                self.TauLimits()



    def GenerateTour(self):
        
        ''' Generates a TSP tour from a randomly 
            selected started node '''

        all_nodes = np.arange(self.n)
        start_node = np.random.choice(all_nodes) # random start node
        current_node = start_node
        tour = [current_node]
        
        for _ in range(self.n-1):

            unvisited_nodes = np.setdiff1d(all_nodes, tour) # index of unvisted nodes

            next_node = self.NextNode(current_node, unvisited_nodes)
            tour.append(next_node)
            current_node = next_node

        tour.append(start_node)

        return tour


    def NextNode(self, current_node, unvisited_nodes):
        
        ''' Performs tournament selection among
            the unvisted nodes '''

        tau_eta = (self.eta[unvisited_nodes, current_node]**self.alpha)*(self.tau[unvisited_nodes, current_node]**self.beta)
        normalization_factor = tau_eta.sum()
        transition_probability = tau_eta / normalization_factor

        r = np.random.rand()
        q = 0

        prob_sum = transition_probability[q]

        while prob_sum <= r:
            q += 1
            prob_sum += transition_probability[q]

        return unvisited_nodes[q]


    def GetPathLength(self, tour):
        return self.D[tour[1:],tour[:-1]].sum()


    def DeltaTauK(self, tour_k, D_k):

        ''' Delta tau matrix for a tour'''

        delta_tau_k = np.zeros((self.n,self.n))
        delta_tau_k[tour_k[1:], tour_k[:-1]] = 1/D_k
        return delta_tau_k
    

    def TauLimits(self):

        ''' Updates tau limits for MMAS '''
        
        self.tau[self.tau < self.tau_min] = float(self.tau_min)
        self.tau[self.tau > self.tau_max] = float(self.tau_max)
        self.tau_max = 1/(self.rho*self.D_best)
        self.tau_min = self.tau_max*((1-0.05**(1/self.n))/((self.n/2)*(0.05**(1/self.n))))


    def PlotTSP(self):

        if self.dim == 2:
            plt.scatter(X[:,0],X[:,1])
            plt.plot(X[self.tour_best[0],0], X[self.tour_best[0],1], 'kx')
            plt.plot(X[self.tour_best,0], X[self.tour_best,1], 'r--')
            plt.show()
        else:
            raise Exception("Plotting is only supported for 2 dimensional data")



if __name__ == '__main__':
    X = np.random.rand(40,2)
    AS = ACO(X=X, mode= "AS")
    AS.Fit(100)
    AS.PlotTSP()

