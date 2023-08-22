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
![image2](https://i.imgur.com/jidYB7d.png)

**LGP example**
![image3](https://i.imgur.com/YIr74AJ.png)
