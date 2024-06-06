import numpy as np
import computeMultFn as cmfn
import scipy.integrate
import integrate
import time

from config import *



# Display program info
print("Simulations with n = {}, n_trials = {}".format(n, n_trials))
print("theta100 = {}".format(theta100))


theta = theta100.astype(int) / 100
nTheta = len(theta)

# Load pre-processing
firstPrimeFactor = np.load("loadfile/firstPrimeFactor_n{}.npy".format(n))
prime = np.load("loadfile/prime_n{}.npy".format(n))
Omega = np.load("loadfile/Omega_n{}.npy".format(n))
sumOfSquare = np.load("loadfile/sumofsquare_theta100_{}_n{}.npy".format("_".join(theta100), n))
print("Loaded: firstPrimeFactor, prime, Omega, sumofsquare")
print("Total number of primes <= {}: {}".format(n, len(prime)))

# Other pre-processing
cMeasure = integrate.normaliseVinfty(theta, prime)

# Uniform([0, 2pi]) angles as values of the angles of Steinhaus mf at prime values
angleSeedFn = lambda x: 2j*np.pi*np.random.random(len(x))

# Memory initialisation
preSteinhaus = np.zeros(n_trials, dtype=complex) # will be used to keep the angles

# results[:, 0, :] = V_n
# results[:, 1, :] = partial sum
results = np.zeros((n_trials, 2, nTheta), dtype=complex)


counter = 0
while True:
    #print("Run {}......".format(counter+1))
    startTime = time.perf_counter()
    for j in range(n_trials):
        print("Run {}, trial {}......".format(counter + 1, j + 1))
        # Generate angles
        preSteinhaus = cmfn.cAddFn(n, angleSeedFn, firstPrimeFactor, displayPerfEverySimulation)
        
        for i in range(nTheta):
            # Evaluate V_n
            intlim = [-100, 100]
            intcounter = time.perf_counter()
            results[j, 0, i] = scipy.integrate.quad(integrate.integrand, intlim[0], intlim[1],
                                                   args = [np.sqrt(theta[i])*np.exp(preSteinhaus[prime.astype(int)]),
                                                           prime, cMeasure[i]],
                                                   limit = 1000)[0]
            if displayPerfEverySimulation:
                print("(n_trials, theta) = ({}, {}): integral computed in {} seconds.".format(
                    j+1, theta[i], time.perf_counter()-intcounter))
                print(results[j, 0, i])

        	# Evaluate twisted Steinhaus and patial sum
            twistedSteinhausFn = np.exp(0.5*np.log(theta[i])*Omega + preSteinhaus)
            results[j, 1, i] = np.sum(twistedSteinhausFn[1:]) / np.sqrt(sumOfSquare[i])
    endTime = time.perf_counter()
    cmfn.printPerformance("Run {}".format(counter+1), endTime-startTime, True)
    for i in range(nTheta):
        np.save("output/point{}/results_n{}_time{}.npy".format(theta100[i], n, int(endTime)), results[:, :, i])
	
	
    print("All results saved.")
    counter += 1
    # Data is saved in a separate document after every n_trials
    # Loop terminates when the experiment has been repeated for a specified number of times
    if counter > 100:
        break




