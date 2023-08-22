# Stochtimizer
Stochastic Optimization Library with;
- Genetic Algorithm (GA)
- Particle Swarm Optimization (PSO)
- Ant Colony Optimization (ACO)
- Linear Genetic Programming (LGP)

---

The workflow is inspired from scikit-learn. Create an object instance of the algorithm and pass a lambda objective function that you want to minimize. Call .fit() on object to start optimization process. 

---

**GA example**

Minimization of objective function $f(x,y) = (x^2+y-11)^2  + (x+y^2-7)^2$. The population consists of 100 binary chromosomes (genes $\in$ {0,1}) with length 16. The first 8 genes are decoded into the x-variable and the remaining 8 for the y-variable, thus each chromosomes defines a point in 2D space. The fitness score for each individual (chromosome) is inverse proportional to their objective function evaluation $f(x_i,y_i)$. The next generation is generated through stochastic mating between individual where more fit indiviuals have higher probability to pass their genome. Here the best individual converged at the point $(x,y) = (3.03,1.87) near one of the 4 minimas where the binary chromosome is shown as $C$ in the figure title. 

![image1](https://i.imgur.com/imquG0K.png)

**ACO example**

Traveling Saleman Problem (TSP) solved using Ant Colony Optimization (ACO).  $n = 30$ randomly generated vertices in the range $(x,y)\in[0,1]\times[0,1]$. $N = 40$ ants (individuals) are used to traverse the graph and the converged solution and shortest found path $D_k = 5.08$ is shown in the figure. Each ant $i$ deposits pheromones along the edges of the graph, with strength inversely proportional to their path distance. The accumulated pheromones for the whole ant colony are stored in a matrix $\tau$. Given a node, ants are more likely to traverse nodes with higher pheromone content. After each epoch the pheromone matrix is updates as $\tau \leftarrow (1-\rho)\tau + \Delta \tau$ where $\rho$ is the evaporation constant and $\Delta \tau$ is the deposited pheromones for the current epoch. 

![image2](https://i.imgur.com/jidYB7d.png)

**LGP example**

Function estimation from $N=20$ data points $(X,y)$ sampled in the range $x\in{0,10}$ of the target function $f = 2x^2+x+3$. Individuals consist of varying length chromosomes that define a sequence of operations on a set of variable register $\mathcal{R}$. Each operation is defined using 4 genes $[g_1,g_2,g_3,g_4]$ where $g_1$ encodes the mathematical operation, $g_2$ encodes which register to write the result to, $g_2,g_3$ encode the operands indices in the variable register. Subsequent generations are generated similar to the process in Genetic Algorithm where fitness score is defined as the $L^2$ loss of the individuals estimated function evaluation $f^*(X) on the data and the target values $y$. 

![image3](https://i.imgur.com/YIr74AJ.png)
