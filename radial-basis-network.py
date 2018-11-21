import scipy as sp
#Technically scipy has all the numpy functionality, but I like the numpy functions better
import numpy as np  
import random

#For visualizing data later
from matplotlib import pyplot as plt


''' 	
	Refer to https://en.wikipedia.org/wiki/Radial_basis_function_network for a quick outline of the network
	structure.
	Quick 101:
		3 layer network with one hidden layer, and an output layer with a radial basis activation function
	Didn't use tensorflow/pytorch to build this because it turns out to be simple enough we don't need to
 '''
class RadialBasisFunction:

	# The radial basis function class has a constructor specifying input/output dimensions, as well as the 
	# number of centers for the gaussian activation.
	def __init__(self, input_dimension, output_dimension, number_centers):
		self.in_dim = input_dimension
		self.out_dim = output_dimension
		self.num_centers = number_centers

		# Parameters of our model initalized randomly of appropriate size
		self.a = self.generateConstantParameters((num_centers,1))
		self.b = self.generateConstantParameters((num_centers,1))
		self.c = self.generateConstantParameters((num_centers,2))


	# Returns the radial activation function: rho_i(|*|)
	def radial_element(x):
		diff = np.subtract(x,c)
		exponential = diff.dot(diff)

		return np.exp(np.abs(b) * exponential)


	# Output of the neural net
	def psi(r):
		return np.sum([a,radial_element(r)],axis=0)


	# Returns uniformly distributed values between 0 and 1 of given shape
	def generateConstantParameters(self, shape):
		return np.random.uniform(0,1,shape)

