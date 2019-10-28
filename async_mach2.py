
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
        self._child = None
        self._session = tf.compat.v1.Session(graph=self._graph)
        with self._graph.as_default():
            self._model_fn()
            self._session.run(tf.compat.v1.global_variables_initializer())

    def train(self, spikes, targets):
        # Computes the output for this node with respect to input spikes.
        # Calculates a target loss. Computes local and downstream gradients from
        # this loss. Send the downstream gradients to children and uses the local
        # gradients to apply a step.
        if self._child:
            # Recursively call children in the graph.
            dspikes = self._child.spike(spikes)
        else:
            dspikes = np.zeros((np.shape(spikes)[0], self._hparams.n_embedding))

        feeds = {
            self._spikes: spikes,
            self._dspikes: dspikes,
            self._targets: targets,
        }
        # Compute the target loss from the input and make a training step.
        run_output = self._session.run([self._tdgrads, self._accuracy, self._tstep], feeds)
        dgrads = run_output[0]
        accuracy = run_output[1]

        if self._child:
            # Recursively pass the gradients through the graph.
            self._child.grade(spikes, dgrads)

        # Return the batch accuracy.
        return accuracy

    def spike(self, spikes):
        # Computes the ouput of this node with respect to input spikes.
        # Returns the embedding produced by this node.
        if self._child:
            # Recursively call the children of this node.
            dspikes = self._child.spike(spikes)
        else:
            # If no children, use dummy inputs for the placeholder.
            dspikes = np.zeros((np.shape(spikes)[0], self._hparams.n_embedding))
        feeds = {
            self._spikes: spikes,
            self._dspikes: dspikes,
        }
        # Run the local graph to retur the embedding. Return the embedding to the
        # parent.
        return self._session.run(self._embedding, feeds)

    def grade(self, spikes, egrads):
        # Computes the gradients for the local node as well as the gradients
        # for its children. Applies the gradients to the local node and sends the
        # children gradients downstream.
        feeds = {
            self._spikes: spikes,
            self._egrads: egrads,
        }
        # Compute gradients for the children and apply the local step.
        dgrads = self.session.run([self._dgrads, self._estep], feeds)[0]
        if self._child:
            # Recursively send the gradiets to the children.
            self._child.grade(spikes, dgrads)

    def _model_fn(self):

        # Placeholders.
        # Spikes: inputs from the dataset of arbitrary batch_size.
        self._spikes = tf.compat.v1.placeholder(tf.float32, [None, self._hparams.n_inputs], 's')
        # Dspikes: inputs from previous component. Size is the same as the embeddings produced
        # by this component.
        self._dspikes = tf.compat.v1.placeholder(tf.float32, [None, self._hparams.n_embedding], 'd')
        # Egrads: Gradient for this components embedding, passed by a parent.
        self._egrads = tf.compat.v1.placeholder(tf.float32, [None, self._hparams.n_embedding], 'g')
        # Targets: Supervised signals used during training and testing.
        self._targets = tf.compat.v1.placeholder(tf.float32, [None, self._hparams.n_targets], 't')

        # Weights and biases.
        # Each component has one hidden layer. Projection is from the input dimension, concatenated
        # with the embeddings from the previous component.
        weights = {
            'w1': tf.Variable(tf.random.truncated_normal([self._hparams.n_inputs + self._hparams.n_embedding, self._hparams.n_hidden], stddev=0.1)),
            'w2': tf.Variable(tf.random.truncated_normal([self._hparams.n_hidden, self._hparams.n_embedding], stddev=0.1)),
            'w3': tf.Variable(tf.random.truncated_normal([self._hparams.n_embedding, self._hparams.n_targets], stddev=0.1)),
        }
        biases = {
            'b1': tf.Variable(tf.constant(0.1, shape=[self._hparams.n_hidden])),
            'b2': tf.Variable(tf.constant(0.1, shape=[self._hparams.n_embedding])),
            'b3': tf.Variable(tf.constant(0.1, shape=[self._hparams.n_targets])),
        }

        # Embedding: Apply the hidden layer to the spikes and embeddings from the previous component.
        # The embedding is the output for this component passed to its parents.
        input_layer = tf.concat([self._spikes, self._dspikes], axis=1)
        hidden_layer = tf.nn.relu(tf.add(tf.matmul(input_layer, weights['w1']), biases['b1']))
        self._embedding = tf.nn.relu(tf.add(tf.matmul(hidden_layer, weights['w2']), biases['b2']))

        # Target: Apply a softmax over the embeddings. This is the loss from the local network.
        # The loss on the target and the loss from the parent averaged.
        logits = tf.add(tf.matmul(self._embedding, weights['w3']), biases['b3'])
        target_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._targets,
                                                              logits=logits))

        # Metrics: We calcuate accuracy here because we are working with MNIST.
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(self._targets, 1))
        self._accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        # Optimizer: The optimizer for this component, could be different accross components.
        optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)

        # Embedding grads: Here, we compute the gradient terms for the embedding with respect
        # to the gradients passed from the parent (a.k.a egrads). Dgrads is the gradient for
        # the downstream component (child) and elgrads are the gradient terms for the the local
        # FFNN.
        self._dgrads = optimizer.compute_gradients(loss=self._embedding, var_list=[self._dspikes], grad_loss=self._egrads)
        self._elgrads = optimizer.compute_gradients(loss=self._embedding, var_list=tf.compat.v1.trainable_variables(), grad_loss=self._egrads)

        # Gradients from target: Here, we computer the gradient terms for the downstream child and
        # the local variables but with respect to the target loss. These get sent downstream and used to
        # optimize the local variables.
        self._tdgrads = optimizer.compute_gradients(loss=target_loss, var_list=[self._dspikes])
        self._tlgrads = optimizer.compute_gradients(loss=target_loss, var_list=tf.compat.v1.trainable_variables())

        # Embedding trainstep: Train step which applies the gradients calculated w.r.t the gradients
        # from a parent.
        self._estep = optimizer.apply_gradients(self._elgrads)

        # Target trainstep: Train step which applies the gradients calculated w.r.t the target loss.
        self._tstep = optimizer.apply_gradients(self._tlgrads)


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
        print (parent.train(batch_x, batch_y))


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
