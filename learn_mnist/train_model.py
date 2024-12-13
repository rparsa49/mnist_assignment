import pickle
import random
from os.path import join
import numpy as np

from learn_mnist.ann import NeuralNet
from learn_mnist.loader import MnistDataloader


input_path = 'data'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')



nn_architecture = [
    {"input_dim": 784, "output_dim": 128, "activation": "sigmoid"},
    {"input_dim": 128, "output_dim": 32, "activation": "sigmoid"},
    {"input_dim": 32, "output_dim": 10, "activation": "sigmoid"},
]


def label_vec(d):
    """
    The label for each image is a number 0-9.

    We represent label i as a vector of length 10 with a 1 in the ith position and zeros elsewhere.
    """
    v = np.zeros((10, 1))
    v[d, 0] = 1.0
    return v


def image_vec(img: np.ndarray):
    """Transform a 28-pixel by 28-pixel image into a column vector of dimension 784 (= 28x28)"""
    return img.reshape((784, 1))



if __name__ == "__main__":
    print("Hi")
    mnist_dataloader = MnistDataloader(
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    )
    (x_train, y_train), _ = mnist_dataloader.load_data()

    num_examples = len(x_train)
    assert len(y_train) == num_examples
    examples = list(range(num_examples))

    ann = NeuralNet(nn_architecture)

    print("Let's train...")
    # Train the neural net on 500_000 minibatches of size 100
    for epoch in range(1, 500_001):
        epoch_examples = random.choices(examples, k=100)
        X = np.hstack([image_vec(x_train[i]) for i in epoch_examples])
        Y = np.hstack([label_vec(y_train[i]) for i in epoch_examples])
        ann.train_batch(X, Y, 0.01)
        if epoch % 1_000 == 0:
            print(f"... finished {epoch} steps ...")
    print("... done.")

    with open("mnist_ann.pickle", "wb") as params_file:
        pickle.dump(ann, params_file)

    print("Saved weights. Have a nice day.")
