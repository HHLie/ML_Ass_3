CSC3022F
ML Assignment 3
LXXHSI007

________________________________________________________________________________
Files:

Part 1

XOR.py -
The handles all input given.
When "train" is given, it will read training data from AND.txt, OR.txt and
NOT.txt files. After reading in the data, it will train 3 perceptrons(AND,OR and
NOT) and validate them.
If "set" is given, perceptrons will be set with pre trained weights.


Perceptron_XOR.py -
Simple Perceptron class, that handles all training, validation and activation
function of the perceptron

Part 2

Classify.py -
Trains a CNN (LeNet-5 model) to classify handwritten digits from the MNIST
dataset. It will train for 6 epochs ~3-5 mins
Handles input of .jpg files and predicts the digit.
The input .jpg file must have a black background and white digit.



Makefile -
A simple makefile with the following commands;
install:
  Installs all packages specified in the requirements.txt file into the virtual
  environment.

venv:
  Check if there is a virtual environment called venv in directory, if there
  isn't one present create one.

clean:
  Remove the virtual environment and delete all .pyc files.

_________________________________________________________________________________________________________
Running the program:
Use "make" or "make install" to create the virtual environment.

Part 1: XOR

python XOR.py

The available input options are:
• train : read training data from AND.txt, OR.txt and NOT.txt files. After
  reading in the data, it will train 3 perceptrons(AND,OR and NOT) and validate
  them.
• set : perceptrons will be set with pre trained weights, from the
  pretrained_weights.txt file.
• save : saves the weights into the pretrained_weights.txt file.
• [input_x1] [input_x2] : Do the XOR function with inputs bound to
  -0.25 < x < 0.25 and 0.75 < x < 1.25. eg. 0.24 0.80


Part 2: Classify

Make sure you have the MNIST dataset in the root folder, and named "MNIST"

python Classify.py

• Run the script to train the data.
• Input the image file path
  - eg. path_to_file/file.jpg



_________________________________________________________________________________________________________
