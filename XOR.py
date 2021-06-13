# ML Assignment 3
# LXXHSI007
# part 1

""" 
Q = (A+B)*[-(A*B)]
XOR = (OR) AND (NAND)
    = (OR) AND NOT(AND)
    
AND gate
I1	I2	O
0   0   0	 	 
0 	1	0
1	0 	0
1	1	1

OR gate
I1	I2	O
0   0   0	 	 
0 	1	1
1	0 	1
1	1	1

NAND gate
I1	I2	1
0   0   1
0 	1	1
1	0 	1
1	1	0

XOR gate
I1	I2	O
0   0   0
0 	1	1
1	0 	1
1	1	0
"""

# packages
import numpy as np
import sys
import os
import random
from Perceptron_XOR import Perceptron


def train(examples, labels, bias, num_inputs, learning_rate, seeded_weights: [float] = None):
    # Create Perceptron
    if seeded_weights is None:
        P = Perceptron(num_inputs, bias=bias)
    else:
        P = Perceptron(num_inputs, bias=bias)

    # print(P.weights)
    valid_percentage = P.validate(examples, labels, verbose=True)
    # print(valid_percentage)
    i = 0
    while valid_percentage < 0.98:  # We want our Perceptron to have an accuracy of at least 80%

        i += 1

        P.train(examples, labels, learning_rate)  # Train our Perceptron
        # print('------ Iteration ' + str(i) + ' ------')
        # print(P.weights)
        valid_percentage = P.validate(examples, labels, verbose=True)  # Validate it
        # print(valid_percentage)

        # This is just to break the training if it takes over 50 iterations. (For demonstration purposes)
        # You shouldn't need to do this as your networks may require much longer to train.
        if i == 50:
            break
    return P


if __name__ == '__main__':
    AND_examples = np.loadtxt("AND.txt", usecols=(0, 1), dtype='float_')
    AND_labels = np.loadtxt("AND.txt", usecols=(2), dtype='float_')
    OR_examples = np.loadtxt("OR.txt", usecols=(0, 1), dtype='float_')
    OR_labels = np.loadtxt("OR.txt", usecols=(2), dtype='float_')
    temp = np.loadtxt("NOT.txt", usecols=(0), dtype='float_')
    NOT_labels = np.loadtxt("NOT.txt", usecols=(1), dtype='float_')
    NOT_examples = []
    for i in temp:
        NOT_examples.append([i])


    print("Training And Gate")
    AND = train(AND_examples, AND_labels, -1, 2, 0.2)
    print("Training OR Gate")
    OR = train(OR_examples, OR_labels, -0.5, 2, 0.2)
    print("Training NOT Gate")
    NOT = train(NOT_examples, NOT_labels, 0.5, 1, 0.2)
    print("---------")
    print("Please enter two inputs or 'exit' to exit:")
    for line in sys.stdin:
        if 'exit' == line.rstrip():
            break
        arr = line.strip().split(" ")
        input_gate = []
        if len(arr) == 2:
            input_gate.append(float(arr[0]))
            input_gate.append(float(arr[1]))
            Or_value = OR.activate(input_gate)
            AND_value = AND.activate(input_gate)
            Nand_value = NOT.activate([AND_value])
            print("XOR Gate:", AND.activate([Or_value, Nand_value]))
        print("Please enter two inputs or 'exit' to exit:")