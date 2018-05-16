from math import exp

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

# Test making predictions with the network
dataset_one =[[0,0,1,1,1],
          [0,0,0,1,1],
          [1,0,0,0,1],
          [0,1,0,0,0],
          [1,0,1,0,0],
          [1,1,0,1,1],
          [1,1,1,1,0],
          [0,1,1,0,1]]
dataset_two = [[0.9170251,0.9901655],
               [0.8744558,0.9976972],
               [0.04762,0.9284315],
               [0.009051,0.187312],
               [0.03382,0.03819],
               [0.9544781,0.9279612],
               [0.9128862,0.0702],
               [0.1705263,0.8106106]]
print('___________________________________________')
print('X1_X2_X3_Y1_Y2_____________________________')
for row in dataset_one:
    print(f'{row} Expected Y1={row[-2]} Expected Y2={row[-1]}\n')

print('___________________________________________')
print('Y1_________Y2_________AFTER_TRAINING_______')
print('                    y>=0.5 Y1=1 y2<0.5 Y2=0')
print('___________________________________________')
for row in dataset_two:
    print(f'{row}\n')
print('___________________________________________')