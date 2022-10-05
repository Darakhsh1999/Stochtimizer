import numpy as np 
from Selectors import TournamentSelection


toursel = TournamentSelection(0.5, 4, replace= False)
toursel.SetSize(10)

fitness = 3*np.arange(10)

while True:
    selection = toursel.Select(fitness)
    print("Selected ind", selection)





