import scipy as sp
#Technically scipy has all the numpy functionality, but I like the numpy functions better
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
		self.out_dim = output_dimension # This is 1, is a probability amplitude
		self.num_centers = number_centers # Number of hidden units

		# Parameters of our model initalized randomly of appropriate size
		self.a = self.generate_constant_parameters((self.num_centers,self.out_dim))
		self.b = self.generate_constant_parameters((self.num_centers,self.out_dim))
		self.c = self.generate_constant_parameters((self.num_centers,self.in_dim))


	# Returns the radial activation function: rho_i(|*|)
	def radial_element(self, x):
		#print(x)
		#print('params', self.a, self.b, self.c[i])
		diff = np.subtract(x.T,self.c)

		norm_array = np.zeros((self.num_centers))
		for i,j in enumerate(diff):
			norm_array[i] = j.dot(j)

		exponential = np.sum(-np.abs(self.b) * norm_array)

		#print('exp', exponential)
		return np.exp(exponential)

	# Update the parameters of the network for training
	# The values for da,db, and dc must be of the correct shape (m x 1) for da/db and (m x 2) for dc
	# Otherwise it will fail silently
	def update_parameters(self, da, db, dc):
		self.a += da
		self.b += db
		self.c += dc


	# Returns uniformly distributed values between 0 and 1 of given shape
	def generate_constant_parameters(self, shape):
		return np.random.uniform(0,1,shape)

	# Operators for stochastic reconfiguration to train neural net
	# In this instance it works better than typical backpropagation 
	# Defines operators to adjust parameters in the neural net according to the formula
	# O_i(n) = d_lambda_i [psi_lambda(n)] / d[psi_lambda(n)]
	
	# r is the domain over which we evaluate psi: (2 x 1) vector here.
	def stochastic_reconfig(self, r):
		psi = self.psi(r)
		
		#O_a operator
		self.o_a = self.radial_element(r) / psi
		
		#O_b operator
		diff = np.subtract(x,c)
		o_b_num = -self.a * self.b * diff.dot(diff) * self.radial_element(r)
		o_b_demon = np.abs(self.b) * psi 
		
		self.o_b = o_b_num / o_b_denom
		
		#O_c operator
		#n_j - c_ij matrix, with some transposes to speed up processing
		intermed_matrix = (self.a - self.c.T).T 
		
		o_c_num = 2 * self.a * np.abs(self.b) * intermed_matrix * self.radial_element(r)
		
		self.o_c = o_c_num / psi
	
	# Output of the neural net, linear combination of the outputs from the hidden layers
	def psi(self,r):
		print ('RTURNED' ,self.radial_element(r))
		return np.sum(self.a, self.radial_element(r), axis=0)
		#self.a[0] * self.radial_element(r,0) + self.a[1] * self.radial_element(r,1) 
