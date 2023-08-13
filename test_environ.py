import numpy as np 

## SELECTION TESTING

#toursel = TournamentSelection(0.5, 4, replace= False)
#toursel.SetSize(10)

#fitness = 3*np.arange(10)

#while True:
    #selection = toursel.Select(fitness)
    #print("Selected ind", selection)

## LAMBA FUN TESTING 

#obj_fun = lambda x,y: 1.5 + x**2 + x*y
#n_vars = obj_fun.__code__.co_argcount
#variable_names = obj_fun.__code__.co_varnames

#X = np.array([[1,1],[1,2],[1,0],[0,5]]) # (n,q)

#variable_dictionary = {}

#for variable_idx, variable_key in enumerate(variable_names):
    #variable_dictionary[variable_key] = X[:,variable_idx]

#x = X[:,0]
#y = X[:,1]

#fitness = obj_fun(*variable_dictionary.values())
#fitness2 = obj_fun(*[x for x in X.T])
#print(fitness)
#print(fitness.shape)
#print(fitness2)
#print(fitness2.shape)

x = np.arange(5)
y = 3*np.sin(x)
import matplotlib.pyplot as plt

plt.plot(x,y)
for i in range(5):
    if i == 4: break
    x_arrow = (x[i]+x[i+1])/2
    y_arrow = (y[i]+y[i+1])/2
    dx_arrow = (x[i+1]-x[i])/4
    dy_arrow = (y[i+1]-y[i])/4
    plt.arrow(x_arrow,y_arrow, dx_arrow, dy_arrow, width=0.01)
plt.show()