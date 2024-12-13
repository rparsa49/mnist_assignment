# Neural Network for Recognizing Handwritten Digits

This project implements a neural network to recognize handwritten digits from the MNIST dataset. The MNIST dataset is a standard benchmarking dataset for computer vision algorithms. Each item in the dataset is a 28x28-pixel grayscale image of a handwritten decimal digit and comes with a label identifying which digit (0-9) is depicted.

The goal is to use supervised learning algorithms to train a model using labeled data and evaluate its ability to recognize digits in new images.

## Installing the Project
From the project directory, create a new Python environment by running:

        python3 -m venv .ven


Activate the environment with:

    source .venv/bin/activate


Install the required packages:

    pip install -r requirements.txt


## Project Contents
	•	data/: Contains the dataset with 60,000 training images and 10,000 test images.
	•	loader.py: Utility class to load the training and test images, along with their labels, from the data files.
	•	try_loader.py: Script to verify that the environment is set up correctly and the data loader is functioning.
	•	train_model.py: Script to create, train, and save the trained model in mnist_ann.pickle.
	•	test_model.py: Script to load the trained model and evaluate its performance on the test dataset.
	•	ann.py: Contains the NeuralNetwork class. This is where the core neural network logic resides, used by the training and testing scripts.

## Viewing the Dataset

To explore the dataset, activate the Python environment and run:

    python -m learn_mnist.try_loader

This script displays random images with their labels from both the training and test sets.

## Training the Model

Train the model by running:

    python -m learn_mnist.train_model

This script will save the trained neural network to mnist_ann.pickle. Note that running this script multiple times will overwrite the file unless you back it up.

## Testing the Model

Test the trained model with:

    python -m learn_mnist.test_model

This script evaluates the model on the test dataset, comparing its predictions to the true labels. Watch the terminal output for results.

## Architecture
The default training script uses a specific network topology with a sigmoid activation function.