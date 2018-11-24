import numpy as np
from random import random, randint

from radial-basis-network import *
from hamiltonian-2d-harm-oss import *

if __name__ == '__main__':

	network = RadialBasisFunctionNetwork(2,1,10)
	ham = Hamiltonian2DOscillator(1,1,0.5)
	# Parameters for 2d oscillator
	energy_x = 4
	energy_y = 2
	
	#Max number of basis functions for hamiltonian
	max_qn = 40
	steps = 200
	
	for i in range(steps):
		
		# Do metropolis to estimate energy
		
		iterations = 50000
		state_new = np.zeros((2,))
		state = np.random.random_sample((2,)) * max
		state_trial = state
		
		accepted_new = 0
		
		#Initialization of energy
		for i in range(iterations / 20):
			#Generate new trial state
			randn = randint(0,1)
			randn2 = (randint(0,1) - 0.5) / 2 #Change state up or down
			state_trial[randn] = state[randn] + randn2 
			
			#Keep states within [0,1] because inputs should be quantum numbers <= max_qn
			state_trial[0] = state_trial[0] ? state_trial[0] : 0
			state_trial[0] = state_trial[0] <= max ? state_trial[0] : 1
			state_trial[1] = state_trial[1] ? state_trial[1] : 0
			state_trial[1] = state_trial[1] <= max ? state_trial[1] : 0
			
			prob = network.psi(state_trial) / network.psi(state)
			
			if random() < np.linalg.norm(prob) ** 2:
				state = state_trial
				accepted_new += 1
				
			
		accepted_new = 0
		energy = 0
		
		#Now do the actual metropolis algorithm
		for i in range(iterations):
			#Generate trial states again
			randn = randint(0,1)
			randn2 = (randint(0,1) - 0.5) / 2
			state_trial[randn] = state[randn] + randn2
			
			state_trial[0] = state_trial[0] ? state_trial[0] : 0
			state_trial[0] = state_trial[0] <= max ? state_trial[0] : 1
			state_trial[1] = state_trial[1] ? state_trial[1] : 0
			state_trial[1] = state_trial[1] <= max ? state_trial[1] : 0

			prob = network.psi(state_trial) / network.psi(state)
			
			#Change state if acceptance probability is high enough
			if random() < np.lingalg.norm(prob) ** 2:
				state = state_trial
				accepted_new += 1
				
	
			#Ground state expectation energy, the first term of E_local
			E = ham.product(state)
			
			#sum over all the states, n'
			state_prime = state

			# The energy calculation depends on if one of the states is the ground state
			# This calculates E
			if not (state[0] or state[1]): #Both dimensions are ground state
				state_prime[0] = state[0] + 1
				coeff1 = np.sqrt(state_prime[0] / 2)
				E += -coeff1 * 
					 energy_x * 
					 network.psi(state_prime) / network.psi(state)
				
				state_prime[1] = state[1] + 1
				coeff2 = np.sqrt(state_prime[1] / 2)
				E += -coeff2 * 
					 energy_y * 
					 network.psi(state_prime) / network.psi(state)

			elif (not state[0] and state[1]):
				state_prime[0] = state[0] + 1
				coeff1 = np.sqrt(state_prime[0] / 2)
				E += -coeff1 *
					 energy_x *
					 network.psi(state_prime) / network.psi(state)

				state_prime = state

				state_prime[1] = state[1] + 1
				coeff2 = np.sqrt(state_prime[1] / 2)

				E += -coeff2 * energy_y * network.psi(state_prime) / network.psi(state)
				state_prime[1] = state[1] - 1
				coeff2 = np.sqrt(state[1] / 2)
				E += -coeff2 * energy_y * network.psi(state_prime) / network.psi(state)

			elif (state[0] and not state[1]):
				state_prime[0] = state[0] + 1
				coeff1 = np.sqrt(state_prime[0] / 2)

				E += -coeff1 * energy_x * network.psi(state_prime) / network.psi(state)  
				state_prime[0] = state[0] - 1 

				coeff1 = np.sqrt(state[0] / 2)

				E += -coeff1 * energy_x * network.psi(state_prime) / network.psi(state)  

				state_prime = state

				state_prime[1] = state[1] + 1
				coeff2 = np.sqrt(state_prime[1] / 2)
				E += -coeff2 * energy_y * network.psi(state_prime) / network.psi(state)  
			else:
				state_prime = state
				state_prime[0] = state[0] + 1 
				coeff1 = np.sqrt(state_prime[0] / 2)

				E += -coeff1 * energy_x * network.psi(state_prime) / network.psi(state)
				state_prime[0] = state[0] - 1 

				coeff1 = np.sqrt(state[0] / 2.0)

				E += -coeff1 * energy_x * network.psi(state_prime) / network.psi(state)

				state_prime = state

				state_prime[1] = state[1] + 1
				coeff2 = np.sqrt(state_prime[1] / 2)

				E += -coeff2 * energy_y * network.psi(state_prime) / network.psi(state)
				state_prime[1] = state[1] - 1
				coeff2 = np.sqrt(state[1] / 2)

				E += -coeff2 * energy_y * network.psi(state_prime) / network.psi(state)


			energy += E

			network.stochastic_reconfig(state)

			parameters = network.o_a + network.o_b + network.o_c