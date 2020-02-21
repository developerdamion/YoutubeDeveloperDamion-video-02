"""
Hey! Thank you for supporting my youtube channel 
( https://www.youtube.com/channel/UCKrs-v_MlKjzVbe4jyLSrPQ ). 

This is video-02: How to build a neural network in under 10 minutes!
Please like and subscribe to my video! 
( https://youtu.be/ybMKuufj9uw )
"""

import numpy as np


class perceptron_nn():

	def __init__(self):
		np.random.seed(1)
		self.synaptic_weights = 2 * np.random.random((3,1)) - 1

	def sigmoid(self, x): return 1 / (1 + np.exp(-x))
	def sigmoid_derivative(self, x): return x * (1-x)

	def test(self, inputs):

		output = np.dot(inputs, self.synaptic_weights)
		output = self.sigmoid(output)
		return output

	def train(self, training_inputs, training_outputs, training_iterations):

		for iteration in range(training_iterations):
			output = self.test(training_inputs)
			error = training_outputs - output
			adjustments = np.dot( training_inputs.T, error * self.sigmoid_derivative(output) )

			self.synaptic_weights += adjustments


if __name__ == "__main__":

	training_inputs = np.array([
									[0,0,1],
									[1,1,1],
									[1,0,1],
									[0,1,1]
											])
	training_outputs = np.array([[0,1,1,0]]).T

	testing_input = np.array([1,1,1])

	perceptron_neural = perceptron_nn()
	perceptron_neural.train(training_inputs, training_outputs, 100000)

	print("synaptic_weights\n", perceptron_neural.synaptic_weights)

	result = perceptron_neural.test(testing_input)

	print("\n",result)