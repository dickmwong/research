import numpy as np

########### Simulation config starts here ##########
#n = 10000000 # 10m
#n = 1000000 # For testing purpose: n = 1m
n = 10000
n_trials = 50
theta100 = np.array(["36", "49", "64", "81", "100"])
displayPerfEverySimulation = False

# Output folders and files
folder = "output/point{}/"
filename = "results_n{}_time{}.npy"

########### Simulation config ends here ##########
