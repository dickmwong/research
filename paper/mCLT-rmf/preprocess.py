import numpy as np
import computeMultFn as cmfn
import time

from config import *


# Output folders and files
prefolder = "loadfile/"
prefilePrimeFactor = "firstPrimeFactor_n{}.npy"
prefilePrime = "prime_n{}.npy"
prefileOmega = "Omega_n{}.npy"
prefileSOS = "sumofsquare_theta100_{}_n{}.npy"


# Display program info
print("Preprocessing for n = {}".format(n))

theta = theta100.astype(int) / 100
nTheta = len(theta)

firstPrimeFactor = cmfn.firstPrime(n, True)
np.save(prefolder+prefilePrimeFactor.format(n), firstPrimeFactor)

prime = np.where(np.arange(n+1) == firstPrimeFactor)[0]
np.save(prefolder+prefilePrime.format(n), prime)

OmegaLambda = (lambda x: 1)
Omega = cmfn.cAddFn(n, OmegaLambda, firstPrimeFactor, True)
np.save(prefolder+prefileOmega.format(n), Omega)

sumOfSquare = np.zeros(nTheta)
for i in range(nTheta):
    sumOfSquare[i] = np.sum(np.power(theta[i], Omega[1:]))
np.save(prefolder+prefileSOS.format("_".join(theta100), n), sumOfSquare)

print("Pre-processing complete!")





