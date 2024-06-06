import numpy as np
import time

def printPerformance(info, totalTime, isPrinted = False):
    if isPrinted:
        print(info + f": computation took {totalTime:.6f} seconds to complete.")
def firstPrime(n, isPrinted=False):
    # returns an array corresponding to the first prime number of n
    # we set the 0th and 1st entries to -1
    # Performance analysis
    startTime = time.perf_counter()

    firstPrimeFactor = np.zeros(n + 1)
    isPrime = np.full(n + 1, True)
    firstPrimeFactor[0:2] = -1
    isPrime[0] = False
    isPrime[1] = False

    for j in range(1, n + 1):
        if isPrime[j] == False:
            pass  # Not prime number
        else:
            firstPrimeFactor[j] = j
            # Prime sieve
            for k in range(2 * j, n + 1, j):
                if isPrime[k] == True:
                    ## First time to come across a prime factor for k
                    firstPrimeFactor[k] = j
                    isPrime[k] = False

    endTime = time.perf_counter()
    totalTime = endTime - startTime
    printPerformance(firstPrime.__name__, totalTime, isPrinted)

    return firstPrimeFactor


def cAddFn(n,  seedFn, firstPrimeFactor=None, isPrinted=False):
    # return an array corresponding to values of a completely multiplicative functions
    # where seedFn is the function evaluation at primes

    if firstPrimeFactor is None:
        firstPrimeFactor = firstPrime(n)
    # Performance analysis
    startTime = time.perf_counter()

    # By default, add[1] = 0
    add = np.zeros(n+1, dtype=complex)

    # Set values of the completely additive function at prime numbers
    isPrime = (firstPrimeFactor == np.arange(n+1))
    add[isPrime] = seedFn(firstPrimeFactor[isPrime].astype(int))


    logn = int(np.ceil(np.log2(n+1)))
    for j in range(2, logn):
        ind_min = int(np.power(2, j))
        ind_max = int(min(np.power(2, j+1), n+1))
        indices = np.arange(ind_min, ind_max)

        firstP = firstPrimeFactor[indices].astype(int)
        remaining = np.floor_divide(indices, firstP)
        add[indices] = add[firstP] + add[remaining]


    endTime = time.perf_counter()
    totalTime = endTime - startTime
    printPerformance(cAddFn.__name__, totalTime, isPrinted)

    return add


def cMultFn(n,  seedFn, firstPrimeFactor=None, isPrinted=False):
    # return an array corresponding to values of a completely multiplicative functions
    # where seedFn is the function evaluation at primes

    if firstPrimeFactor is None:
        firstPrimeFactor = firstPrime(n)
    # Performance analysis
    startTime = time.perf_counter()

    # By default, mult[1] = 0
    mult = np.ones(n+1, dtype=complex)

    # Set values of the completely additive function at prime numbers
    isPrime = (firstPrimeFactor == np.arange(n+1))
    mult[isPrime] = seedFn(firstPrimeFactor[isPrime].astype(int))


    logn = int(np.ceil(np.log2(n+1)))
    for j in range(2, logn):
        ind_min = int(np.power(2, j))
        ind_max = int(min(np.power(2, j+1), n+1))
        indices = np.arange(ind_min, ind_max)

        firstP = firstPrimeFactor[indices].astype(int)
        remaining = np.floor_divide(indices, firstP)
        mult[indices] = mult[firstP] * mult[remaining]


    endTime = time.perf_counter()
    totalTime = endTime - startTime
    printPerformance(cMultFn.__name__, totalTime, isPrinted)

    return mult





