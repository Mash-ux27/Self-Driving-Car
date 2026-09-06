import numpy as np


# Represents a small feedforward neural network controlled by genome weights.
class Brain:
	def __init__(self, input_size=5, hidden_size=32, second_hidden_size=32, output_size=2):
		# Create random weights and biases for both 32-neuron hidden layers and outputs.
		self.input_weights = np.random.uniform(-1, 1, (input_size, hidden_size))
		self.first_hidden_bias = np.random.uniform(-1, 1, hidden_size)
		self.second_layer_weights = np.random.uniform(-1, 1, (hidden_size, second_hidden_size))
		self.second_hidden_bias = np.random.uniform(-1, 1, second_hidden_size)
		self.output_weights = np.random.uniform(-1, 1, (second_hidden_size, output_size))
		self.output_bias = np.random.uniform(-1, 1, output_size)

	# Process sensor inputs through two hidden layers into steering and throttle signals.
	def forward(self, inputs):
		# Apply the first hidden layer to the five normalized sensor values.
		first_hidden = np.tanh(np.dot(inputs, self.input_weights) + self.first_hidden_bias)
		# Process those activations through the second hidden layer.
		second_hidden = np.tanh(np.dot(first_hidden, self.second_layer_weights) + self.second_hidden_bias)
		# Return steering and throttle signals in the range -1 to 1.
		return np.tanh(np.dot(second_hidden, self.output_weights) + self.output_bias)

	# Flatten every weight and bias into one vector for genetic operations.
	def genome(self):
		return np.concatenate([
			self.input_weights.flatten(),
			self.first_hidden_bias.flatten(),
			self.second_layer_weights.flatten(),
			self.second_hidden_bias.flatten(),
			self.output_weights.flatten(),
			self.output_bias.flatten(),
		])

	# Build a brain by reshaping a flat genome back into network parameters.
	@classmethod
	def from_genome(cls, genome, input_size=5, hidden_size=32, second_hidden_size=32, output_size=2):
		brain = cls(input_size, hidden_size, second_hidden_size, output_size)
		position = 0

		input_count = input_size * hidden_size
		brain.input_weights = genome[position:position + input_count].reshape(input_size, hidden_size)
		position += input_count

		brain.first_hidden_bias = genome[position:position + hidden_size]
		position += hidden_size

		second_count = hidden_size * second_hidden_size
		brain.second_layer_weights = genome[position:position + second_count].reshape(hidden_size, second_hidden_size)
		position += second_count

		brain.second_hidden_bias = genome[position:position + second_hidden_size]
		position += second_hidden_size

		output_count = second_hidden_size * output_size
		brain.output_weights = genome[position:position + output_count].reshape(second_hidden_size, output_size)
		position += output_count

		brain.output_bias = genome[position:position + output_size]
		return brain
