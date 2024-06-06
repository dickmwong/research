import numpy as np
#import scipy.integrate

def normaliseVinfty(thetas, primes):
	# Returns the factor that should be multiplied to the random Euler product
	# which is needed for the "correct normalisation" for V_infty.
	# Note: this factor includes the division of 2pi
	primeDivtheta = np.outer(1/thetas, primes)
	return np.prod(1 - 1/primeDivtheta, axis = 1) / (2 * np.pi)
	


def integrand(x, args):
	# define the measure m_n(dx) for integration
	# args[0]: mf f(p) alpha(p)
	# args[1]: list of prime values
	# args[2]: normaliseVinfty(theta, primes); this value should be pre-computed for optimal performance
	return args[2] * np.prod(np.abs(1 - args[0] * np.power(args[1], -0.5 - 1.0j * x))** -2) / (0.25 + x**2)


