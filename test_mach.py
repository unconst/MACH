from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy
import queue

import mach

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

if __name__ == "__main__":
    c = mach.Config()
    m = mach.Mach(c, mnist)

    for i in range(1000):
        m.Train(1)
        m.Learn(1)
        if i % 100 == 0:
            m.Test()
