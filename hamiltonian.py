#Add some more here
import numpy as np
class Hamiltonian2DOscillator:

	def __init__(self, omega, m, lam):
		self.omega = omega
		self.mass = m
		self.lam = lam
		
		
	#Computes the inner product <n|H|n>
	# where |n> is a 2 component wavefunction
	def product(self, n):
		return np.sum((n + 0.5) * self.omega)
		