#Add some more here
import numpy as np
import copy
class Hamiltonian2DOscillator:

	def __init__(self, omega, m, lam, ex, ey):
		self.omega = omega
		self.mass = m
		self.lam = lam
		self.energy_x = ex
		self.energy_y = ey
		
		
	# Computes the inner product <n|H|n>
	# Returns the eigenvalue of the unperturbed hamiltonian
	def product(self, n):
		return np.sum((n + 0.5) * self.omega, dtype=np.complex)
		

	# Computes the inner product <n|H'|n>
	# Returns the component of energy from the hamiltonian pertubation 
	def perturbed_energy(self, state, network):

		state_prime = copy.deepcopy(state)

		E = 0

		# Calculation differs because of boundary states and ladder operators
		if (state[0] == 0 and state[1] == 0): #Both dimensions are ground state
			state_prime[0] = state[0] + 1
			coeff1 = np.sqrt(state_prime[0] / 2)
			E += -coeff1 * self.energy_x * network.psi(state_prime) / network.psi(state)
			
			state_prime = copy.deepcopy(state)

			state_prime[1] = state[1] + 1
			coeff2 = np.sqrt(state_prime[1] / 2)
			E += -coeff2 * self.energy_y * network.psi(state_prime) / network.psi(state)

		elif (state[0] == 0 and state[1]):
			state_prime[0] = state[0] + 1
			coeff1 = np.sqrt(state_prime[0] / 2)
			E += -coeff1 * self.energy_x *network.psi(state_prime) / network.psi(state)

			state_prime = copy.deepcopy(state)

			state_prime[1] = state[1] + 1
			coeff2 = np.sqrt(state_prime[1] / 2)

			E += -coeff2 * self.energy_y * network.psi(state_prime) / network.psi(state)
			state_prime[1] = state[1] - 1
			coeff2 = np.sqrt(state[1] / 2)
			E += -coeff2 * self.energy_y * network.psi(state_prime) / network.psi(state)

		elif (state[0] and state[1] == 0):
			state_prime[0] = state[0] + 1
			coeff1 = np.sqrt(state_prime[0] / 2)

			E += -coeff1 * self.energy_x * network.psi(state_prime) / network.psi(state)  
			state_prime[0] = state[0] - 1 

			coeff1 = np.sqrt(state[0] / 2)

			E += -coeff1 * self.energy_x * network.psi(state_prime) / network.psi(state)  

			state_prime = copy.deepcopy(state)

			state_prime[1] = state[1] + 1
			coeff2 = np.sqrt(state_prime[1] / 2)
			E += -coeff2 * self.energy_y * network.psi(state_prime) / network.psi(state)  
		else:
			state_prime[0] = state[0] + 1 
			coeff1 = np.sqrt(state_prime[0] / 2)

			E += -coeff1 * self.energy_x * network.psi(state_prime) / network.psi(state)
			state_prime[0] = state[0] - 1 

			coeff1 = np.sqrt(state[0] / 2.0)

			E += -coeff1 * self.energy_x * network.psi(state_prime) / network.psi(state)

			state_prime = copy.deepcopy(state)

			state_prime[1] = state[1] + 1
			coeff2 = np.sqrt(state_prime[1] / 2)

			E += -coeff2 * self.energy_y * network.psi(state_prime) / network.psi(state)
			state_prime[1] = state[1] - 1
			coeff2 = np.sqrt(state[1] / 2)

			E += -coeff2 * self.energy_y * network.psi(state_prime) / network.psi(state)

		return E