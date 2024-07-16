import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 

# Read training data from csv 
AND_data = pd.read_csv ('./AND.csv')

# print (AND_data)
'''
   x1  x2  y
0   0   0  0
1   0   1  0
2   1   0  0
3   1   1  1

'''
# Extract x1, x2, and y values from CSV
x1 = np.array (AND_data["x1"])
x2 = np.array (AND_data["x2"])
y =  np.array (AND_data["y"])

# Combine x1, x2, and y into a single array
data = np.stack((x1, x2, y), axis=1)  # axis : 0 表示 row 軸 ( 預設值 )，1 為 column 軸

'''
x1 : [0, 0, 1, 1]
x2 : [0, 1, 0, 1]
y : [0, 0, 0, 1]


merge x1, x2, y
[[0 0 0]
 [0 1 0]
 [1 0 0]
 [1 1 1]]

'''

# Initialize the bias parameter and the weights
bias = 0
w1 = 0
w2 = 0
lr = 0.1
num_epochs = 30
errors = []
outputs = []


# Training the perceptron
for epoch in range (num_epochs):
    print(f"\nEpoch: {epoch + 1}")
    total_error = 0

    for i in range (len(data)):
        x1 = data[i][0]
        x2 = data[i][1]
        y = data[i][2]
        y_pred = x1 * w1 + x2 * w2 + bias

        # Update prediction: output > 0.5 represent 1 , otherwise 0
        y_pred = (y_pred > 0.5) * 1.0
        outputs.append(y_pred)

        # Caculate error
        error = y - y_pred
        total_error += error

        # Update the weights and bias
        w1 += lr * error * x1
        w2 += lr * error * x2
        bias += lr * error
        
        print(f"w1: {w1:.6f}, w2: {w2:.6f}, bias: {bias:.6f}")
    
    print ("Output from perceptron ", outputs)

    # Clear output to record the next epoch
    outputs = []
        
    # Record error for each epoch
    errors.append(total_error)        


# Plot the training loss
plt.figure(0)
plt.plot(errors)
plt.xlabel("Epochs")
plt.ylabel("Loss value")
plt.title("Training Loss Over Epochs")
plt.show()
