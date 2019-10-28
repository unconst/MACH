
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
        self._hparams = hparams
        self._graph = tf.Graph()
        self._session = tf.compat.v1.Session(graph=self._graph)
        with self._graph.as_default():
            self._model_fn()
            self._session.run(tf.global_variables_initializer())

    def embedding(self, spikes, dspikes):
        feeds = {
            self._spikes: spikes,
            self._dspikes: dspikes,
        }
        return self._session.run(self._embedding, feeds)


    def get_embedding(self, spikes, dspikes = None):
        if dspikes == None:
            dspikes = np.zeros((np.shape(spikes)[0], self._hparams.n_embedding))
        feeds = {
            self._spikes: spikes,
            self._dspikes: dspikes,
        }
        return self._session.run(self._embedding, feeds)

    def grade(self, spikes, egrads):
        feeds = {
            self._spikes: spikes,
            self._egrads: egrads,
        }
        dgrads = self.session.run([self._step, self._dgrads], feeds)[1]
        if self._child:
            self._child.grade(spikes, dgrads)

    def _embedding_fn(self):

        # Placeholders.
        self._spikes = tf.placeholder(tf.float32, [None, self._hparams.n_inputs], 'x')
        self._dspikes = tf.placeholder(tf.float32, [None, self._hparams.n_embedding], 'd')
        self._egrads = tf.placeholder(tf.float32, [None, self._hparams.n_embedding], 'g')

        # Weights and biases.
        weights = {
            'w1': tf.Variable(tf.truncated_normal([self._hparams.n_inputs + self._hparams.n_embedding, self._hparams.n_hidden], stddev=0.1)),
            'w2': tf.Variable(tf.truncated_normal([self._hparams.n_hidden, self._hparams.n_embedding], stddev=0.1)),
        }
        biases = {
            'b1': tf.Variable(tf.constant(0.1, shape=[self._hparams.n_hidden])),
            'b2': tf.Variable(tf.constant(0.1, shape=[self._hparams.n_embedding])),
        }
        hvars = [weights['w1'], weights['w2'], biases['b1'], biases['b2']]

        # Embedding.
        input_layer = tf.concat([self._spikes, self._dspikes], axis=1)
        hidden_layer = tf.nn.relu(tf.add(tf.matmul(input_layer, weights['w1']), biases['b1']))
        self._embedding = tf.nn.relu(tf.add(tf.matmul(hidden_layer, weights['w2']), biases['b2']))

        # Gradients.
        self._dgrads = tf.gradients(ys=self._embedding, xs=[self._dspikes], grad_ys=self._egrads)
        self._hgrads = tf.gradients(ys=self._embedding, xs=hvars, grad_ys=self._egrads)

    def _target_fn(self):

        # Placeholders.
        self._targets = tf.placeholder(tf.float32, [None, self._hparams.n_targets], 'x')

        # Weights and biases.
        weights = {
            'w1': tf.Variable(tf.truncated_normal([self._hparams.n_embedding, self._hparams.n_targets], stddev=0.1)),
        }
        biases = {
            'b1': tf.Variable(tf.constant(0.1, shape=[self._hparams.n_targets])),
        }
        tvars = [weights['w1'], weights['w2'], biases['b1'], biases['b2']]

        logits = tf.add(tf.matmul(self._embedding, weights['w1']), biases['b1'])

        target_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._targets,
                                                               logits=logits))

        # Gradients.
        self._tegrads = tf.gradients(ys=target_loss, xs=[self._embedding])
        self._tgrads = tf.gradients(ys=target_loss, xs=tvars)

    def _htrain_fn(self):

        self._hgrad_placeholders = []
        for grad_var in self._hgrads:
            placeholder = tf.placeholder( 'float', shape=grad_var[1].get_shape())
            self._hgrad_placeholders.append((placeholder, grad_var[1]))

        # Optimizer.
        optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)

        # hidden train step.
        self._htrain_step = optimizer.apply_gradients(self._hgrad_placeholders)

    def _ttrain_fn(self):

        self._tgrad_placeholders = []
        for grad_var in self._tgrads:
            placeholder = tf.placeholder( 'float', shape=grad_var[1].get_shape())
            self._tgrad_placeholders.append((placeholder, grad_var[1]))

        # Optimizer.
        optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)

        # hidden train step.
        self._ttrain_step = optimizer.apply_gradients(self._tgrad_placeholders)

    def _htrain_fn(self):

        self._hgrad_placeholders = []
        for grad_var in self._hgrads:
            placeholder = tf.placeholder( 'float', shape=grad_var[1].get_shape())
            self._hgrad_placeholders.append((placeholder, grad_var[1]))

        # Optimizer.
        optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)

        # hidden train step.
        self._htrain_step = optimizer.apply_gradients(self._hgrad_placeholders)


    def _ltrain_fn(self):

        self._lgrad_placeholders = []
        for grad_var in self._lgrads:
            placeholder = tf.placeholder( 'float', shape=grad_var[1].get_shape())
            self._lgrad_placeholders.append((placeholder, grad_var[1]))

        # Optimizer.
        optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)

        self._ltrain_step = optimizer.apply_gradients(self._lgrad_placeholders)

        self._spikes = tf.placeholder(tf.float32, [None, self._hparams.n_inputs], 'x')


        # Optimizer.
        optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)

        # Step.
        self._step = optimizer.apply_gradients(lgrads)

        self._logits = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
        target_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._targets,
                                                       logits=self._logits))

        # Optimizer.
        optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)

        # Gradients
        self._ugrads = tf.placeholder(tf.float32, [None, self._hparams.n_embedding], 'g')
        self._dgrads = optimizer.compute_gradients(loss=self._embedding, var_list=[self._dspikes], grad_loss=self._ugrads)
        lgrads = optimizer.compute_gradients(loss=self._embedding, grad_loss=self._ugrads)

        # Step.
        self._step = optimizer.apply_gradients(lgrads)



def main(hparams):

    mnist, hparams = load_data_and_constants(hparams)

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
    parser.add_argument(
        '--n_embedding',
        default=128,
        type=int,
        help='Size of embedding between components. Default n_embedding=128')
    parser.add_argument(
        '--n_components',
        default=2,
        type=int,
        help='The number of training iterations. Default n_components=2')
    parser.add_argument(
        '--n_iterations',
        default=10000,
        type=int,
        help='The number of training iterations. Default n_iterations=10000')
    parser.add_argument(
        '--n_hidden',
        default=512,
        type=int,
        help='Size of layer 1. Default n_hidden=512')
    parser.add_argument(
        '--n_print',
        default=100,
        type=int,
        help=
        'The number of iterations between print statements. Default n_print=100'
    )

    hparams = parser.parse_args()

    main(hparams)
