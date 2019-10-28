
import os
import sys
import time
import argparse
from loguru import logger
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def load_data_and_constants(hparams):
    '''Returns the dataset and sets hparams.n_inputs and hparamsn_targets.'''
    # Load mnist data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    hparams.n_inputs = 784
    hparams.n_targets = 10
    return mnist, hparams

class Mach:

    def __init__(self, hparams):
        self._child = None
        self._hparams = hparams
        self._graph = tf.Graph()
        self._session = tf.compat.v1.Session(graph=self._graph)
        with self._graph.as_default():
            self._model_fn()
            self._session.run(tf.global_variables_initializer())

    def spike(self, spikes):
        if self._child:
            dspikes = self._child.spike(spikes)
        else:
            dspikes = numpy.zeros((numpy.shape(spikes)[0], N_EMBEDDING))
        feeds = {
            self._spikes: spikes,
            self._dspikes: dspikes,
        }
        return self._session.run(self.embedding, feeds)

    def grade(self, spikes, ugrads):
        feeds = {
            self._spikes: spikes,
            self._ugrads: ugrads,
        }
        dgrads = self.session.run([self._step, self._dgrads], feeds)[1]
        if self._child:
            self._child.grade(spikes, dgrads)

    def _model_fn(self):

        self._spikes = tf.placeholder(tf.float32, [None, self._hparams.n_inputs], 'x')
        self._dspikes = tf.placeholder(tf.float32, [None, self._hparams.embedding_size], 'y')

        weights = {
            'w1': tf.Variable(tf.truncated_normal([self._hparams.n_inputs + self._hparams.n_embedding, self._hparams.n_layer1], stddev=0.1)),
            'w2': tf.Variable(tf.truncated_normal([512, self._hparams.embedding_size], stddev=0.1)),
            'w3': tf.Variable(tf.truncated_normal([128, 10], stddev=0.1)),
        }
        biases = {
            'b1': tf.Variable(tf.constant(0.1, shape=[512])),
            'b2': tf.Variable(tf.constant(0.1, shape=[128])),
            'b3': tf.Variable(tf.constant(0.1, shape=[10])),
        }

        # FFNN
        input_layer = tf.concat([self._spikes, self._dspikes], axis=1)
        layer_1 = tf.nn.relu(tf.add(tf.matmul(input_layer, weights['w1']), biases['b1']))
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['w2']), biases['b2']))
        self._embedding = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])

        # Optimizer.
        optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)

        # Gradients
        self._ugrads = tf.placeholder(tf.float32, [None, N_EMBEDDING], 'y')
        self._dgrads = optimizer.compute_gradients(loss=self._embedding, var_list=[self._dspikes], grad_loss=self._ugrads)
        lgrads = optimizer.compute_gradients(loss=self._embedding, grad_loss=self._ugrads)

        # Step.
        self._step = optimizer.apply_gradients(lgrads)



def main(hparams):

    load_data_and_constants(hparams)

    # Build async components.
    components = []
    for i in range(hparams.n_components):
        mach = Mach(hparams)
        if i != 0:
            mach.child = components[i-1]
        components.append(mach)




    # Training loop.
    parent = components[hparams.n_components-1]
    for i in range(hparams.n_iterations):
        batch_x, batch_y = mnist.train.next_batch(hparams.batch_size)
        print (parent.spike(batch_x, batch_y))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch_size',
        default=128,
        type=int,
        help='The number of examples per batch. Default batch_size=128')
    arser.add_argument(
        '--n_embedding',
        default=128,
        type=int,
        help='Size of embedding between components. Default n_embedding=128')
    parser.add_argument(
        '--n_components',
        default=2,
        type=int,
        help='The number of training iterations. Default n_iterations=10000')
    parser.add_argument(
        '--n_iterations',
        default=10000,
        type=int,
        help='The number of training iterations. Default n_iterations=10000')
    parser.add_argument(
        '--n_print',
        default=100,
        type=int,
        help=
        'The number of iterations between print statements. Default n_print=100'
    )

    hparams = parser.parse_args()

    main(hparams)
