import pickle
from os.path import join

import numpy as np

from learn_mnist.ann import NeuralNet
from learn_mnist.loader import MnistDataloader

input_path = 'data'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')



def image_vec(img: np.ndarray):
    return img.reshape((784, 1))

def label_from_vec(v: np.ndarray) -> int:
    max = -99999999999.0
    idx = -1
    for i, x in enumerate(v):
        if x[0] > max:
            idx = i
            max = x[0]
    return idx


if __name__ == "__main__":
    print("Hi there!")

    with open("mnist_ann.pickle", "rb") as picklefile:
        ann: NeuralNet = pickle.load(picklefile)


    mnist_dataloader = MnistDataloader(
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    )
    _, (x_test, y_test) = mnist_dataloader.load_data()

    num_examples = len(x_test)
    assert len(y_test) == num_examples
    examples = list(range(num_examples))
    wrong_examples = set()
    print("Example\tExpected\tGot")
    print("===================================")
    for example in examples:
        y_hat = ann.apply(image_vec(x_test[example]))
        y_hat_label = label_from_vec(y_hat)
        y_label = y_test[example]
        if y_hat_label != y_label:
            wrong_examples.add(example)
            wrong = True
        else:
            wrong = False
        print(f"{example}\t\t{y_label}\t\t\t{y_hat_label}\t\t{'X' if wrong else ''}")

    print(f"The model got {len(wrong_examples)} test examples out {len(examples)} of  wrong.")

