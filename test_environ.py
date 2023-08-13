import numpy as np 
from selectors import TournamentSelection


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

N = 11
ptr = 0
for i in range (ptr,N-1,2):
    print(i,i+1)