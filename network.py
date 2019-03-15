
import numpy as np  
import random

#For visualizing data later
#from matplotlib import pyplot as plt


''' 	
	Refer to https://en.wikipedia.org/wiki/Radial_basis_function_network for a quick outline of the network
	structure.
	Quick 101:
		3 layer network with one hidden layer, and an output layer with a radial basis activation function
	Didn't use tensorflow/pytorch to build this because it turns out to be simple enough we don't need to
 '''
class RadialBasisFunctionNetwork:

	# The radial basis function class has a constructor specifying input/output dimensions, as well as the 
	# number of centers for the gaussian activation.
	def __init__(self, input_dimension, output_dimension, number_centers):
		self.in_dim = input_dimension # This is 2 for a 2D system
		self.out_dim = output_dimension # This is always 1, is a probability amplitude
		self.num_centers = number_centers # Number of hidden units

		# Parameters of our model initalized randomly of appropriate size
		self.a = self.generate_constant_parameters((self.num_centers,self.out_dim))
		#self.b = self.generate_constant_parameters((self.num_centers,self.out_dim))
		self.b = np.ones((self.num_centers, self.out_dim), dtype=np.complex)
		self.c = self.generate_constant_parameters((self.num_centers,self.in_dim))


	# Returns the radial activation function: rho_i(|*|)
	def radial_element(self, x):
		#print(x)
		#print('params', self.a, self.b, self.c[i])
		diff = np.zeros((self.num_centers, self.in_dim), dtype=np.complex)

		diff += np.subtract(x.T,self.c)

		norm_array = np.zeros((self.num_centers),dtype=np.complex)
		for i,j in enumerate(diff):
			norm_array[i] = j.dot(j)

		exp = -np.abs(self.b)*(np.reshape(norm_array, (10,1)))
		#print('exponential factor', exp)

		exponential = exp.astype(np.complex)

		#print('expoential', exponential)
		#print('exp', exponential)
		for i,j in enumerate(exponential):
			if j < -400:
				exponential[i] = 0

		return np.exp(exponential).astype(np.complex)

	# Update the parameters of the network for training
	# The values for da,db, and dc must be of the correct shape (m x 1) for da/db and (m x 2) for dc
	# Otherwise it will fail silently
	def update_parameters(self, da, db, dc):
		self.a += da
		self.b += db
		self.c += dc


	# Returns uniformly distributed values between 0 and 1 of given shape
	def generate_constant_parameters(self, shape):
		#return np.ones(shape,dtype=np.complex) / 2
		return np.random.uniform(0,1,shape).astype(np.complex)

	# Operators for stochastic reconfiguration to train neural net
	# In this instance it works better than typical backpropagation 
	# Defines operators to adjust parameters in the neural net according to the formula
	# O_i(n) = d_lambda_i [psi_lambda(n)] / d[psi_lambda(n)]
	
	# r is the domain over which we evaluate psi: (2 x 1) vector here.
	def stochastic_reconfig(self, r):
		psi = self.psi(r).astype(np.complex)
		
		#O_a operator
		self.o_a = self.radial_element(r) / psi

		temp = []
		#O_b operator
		for i in range(self.num_centers):
			diff = np.subtract(r.T,self.c[i])
			temp.append(diff.dot(diff))
		
		temp = np.reshape(np.array(temp), np.shape(self.a))
		#print(self.a * temp)
		#print(self.a, self.b, temp, self.radial_element(r))
		try:
			o_b_num = -self.a * self.b * temp * self.radial_element(r)
		except:
			print(self.a)
			print(self.b)
			print(temp)
			print(self.radial_element)

		o_b_denom = np.abs(self.b) * psi 
		
		#print('num', o_b_num)
		#print('denom', o_b_denom)
		self.o_b = o_b_num / o_b_denom
		
		#O_c operator

		o_c_num = np.zeros((self.in_dim, self.num_centers), dtype=np.complex)
		intermed = []
		for k in range(self.num_centers):
			
			intermed.append(np.subtract(r.T, self.c[k]))

		intermed = np.reshape(np.array(intermed), (self.num_centers, self.in_dim))

		#print('int', intermed_matrix)
		#print('intermediate', 2 * self.a * np.abs(self.b) * intermed_matrix * self.radial_element(r))
		o_c_num = 2 * self.a * np.abs(self.b) * intermed * self.radial_element(r)
		
		#print('after', np.shape(o_c_num[:,0]))

		temp = np.zeros((self.in_dim * self.num_centers), dtype=np.complex)
		for i in range(self.in_dim):
			temp[self.num_centers * i: self.num_centers * (i+1)] = o_c_num[:,i]

		self.o_c = np.reshape(temp / psi , (self.in_dim * self.num_centers, 1))
	
	# Output of the neural net, linear combination of the outputs from the hidden layers
	def psi(self,r):
		#print ('RTURNED' ,self.radial_element(r))
		temp = self.radial_element(r)
		radial_ele = np.zeros((np.shape(temp)), dtype=np.complex)

		for i,j in enumerate(temp):
			radial_ele[i] = 0 if j == 1 else temp[i]

		return np.sum(self.a * self.radial_element(r))
		#self.a[0] * self.radial_element(r,0) + self.a[1] * self.radial_element(r,1) 
		
if __name__ == '__main__':
	network = RadialBasisFunctionNetwork(2,1,10)

	#print(network.psi(np.array([2,2])))
	network.stochastic_reconfig(np.array([2,2]))

	#print(network.o_a)
	#print(network.o_b)
	#print(network.o_c)
