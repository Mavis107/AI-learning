import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 

# read training data from csv 
XOR_data = pd.read_csv ('./XOR.csv')

'''
   x1  x2  y
0   0   0  0
1   0   1  1
2   1   0  1
3   1   1  0

'''

# Get x1, x2 value from csv 
x1 = np.array (XOR_data["x1"])
x2 = np.array (XOR_data["x2"])
y =  np.array (XOR_data["y"]).reshape(-1, 1)  # Ensure y is a column vector)

'''
[0 1 1 0]

to

[[0]
 [1]
 [1]
 [0]]

'''

# Stack x1 and x2 to create the input matrix data
data = np.stack((x1, x2), axis=1)

# Neural network parameters
input_layer_neurons = data.shape[1]    # Number of features in input data
hidden_layer_neurons = 2               # Number of hidden layer neurons
output_neurons = 1                     # Number of output neurons
lr = 0.1                               # Learning rate
momentum = 0.9                         # Momentum
epochs = 5000                          # Number of training epochs


# Weight initialization
np.random.seed(0)  # For reproducibility
hidden_weights = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
hidden_bias = np.random.uniform(size=(1, hidden_layer_neurons))
output_weights = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
output_bias = np.random.uniform(size=(1, output_neurons))

print("Initial hidden weights:\n", hidden_weights)
print("Initial hidden biases:\n", hidden_bias)
print("Initial output weights:\n", output_weights)
print("Initial output biases:\n", output_bias)


# Initialize velocity for momentum
v_hidden_weights = np.zeros_like(hidden_weights)
v_hidden_bias = np.zeros_like(hidden_bias)
v_output_weights = np.zeros_like(output_weights)
v_output_bias = np.zeros_like(output_bias)


# sigmoid function
def sigmoid (x) :
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


# Training the neural network
losses = []
for epoch in range (epochs) :
	# Forward Propagation
    hidden_layer_input = np.dot (data, hidden_weights) + hidden_bias
    hidden_layer_activation = sigmoid (hidden_layer_input)

    output_layer = np.dot (hidden_layer_activation, output_weights) + output_bias
    predicted_output = sigmoid (output_layer)

    # Backpropagation
    error = y - predicted_output
    d_predicted_output = error * sigmoid_derivative (predicted_output)
    
    error_hidden_layer = d_predicted_output.dot (output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative (hidden_layer_activation)
  
    # Updating weights and biases velocities
    v_output_weights = momentum * v_output_weights + hidden_layer_activation.T.dot(d_predicted_output) * lr 
    v_output_bias = momentum * v_output_bias + np.sum (d_predicted_output, axis = 0, keepdims = True) * lr
    v_hidden_weights = momentum * v_hidden_weights + data.T.dot (d_hidden_layer) * lr
    v_hidden_bias = momentum * v_hidden_bias + np.sum(d_hidden_layer, axis = 0, keepdims = True) * lr

    # Updating weights and biases
    output_weights += v_output_weights
    output_bias += v_output_bias
    hidden_weights += v_hidden_weights
    hidden_bias += v_hidden_bias

    # Record the loss for each epoch
    loss = np.mean(np.square(error))
    losses.append(loss)

    # Print loss for every 1000 epochs
    if (epoch+1) % 1000 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss:.6f}')


# Print final weights and biases
print("\nFinal weights and biases after training:")
print("Weights between input and hidden layer:\n", hidden_weights)
print("Biases of hidden layer:\n", hidden_bias)
print("Weights between hidden and output layer:\n", output_weights)
print("Biases of output layer:\n", output_bias)

# Update prediction: output > 0.5 represent 1 , otherwise 0
predicted_output = (predicted_output > 0.5) * 1.0
print("Predict output after 5000 epochs:\n", predicted_output)


# Plot the training loss
plt.figure(0)
plt.plot(range(epochs), losses)
plt.xlabel("Epochs")
plt.ylabel("Loss value")
plt.title("Training Loss Over Epochs")
plt.show()
