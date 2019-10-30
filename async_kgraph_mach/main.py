import os
import sys
import time
import argparse
import random
from loguru import logger
import numpy as np
import tensorflow as tf
import threading
from tensorflow.examples.tutorials.mnist import input_data


def load_data_and_constants(hparams):
    '''Returns the dataset and sets hparams.n_inputs and hparamsn_targets.'''
    # Load mnist data
    mnist = input_data.read_data_sets("~/tmp/MNIST_data/", one_hot=True)
    hparams.n_inputs = 784
    hparams.n_targets = 10
    return mnist, hparams


def next_nounce():
    # Random number from large range.
    return random.randint(0, 1000000000)


class MACH:

    def __init__(self, name, hparams):

        self.name = name
        self._mnist, self._hparams = load_data_and_constants(hparams)
        self._mem = {}
        self._children = [None for _ in range(self._hparams.n_children)]
        self._ema_deltaLij = [0.0 for _ in range(self._hparams.n_children)]
        self._graph = tf.Graph()
        self._file_writer = tf.compat.v1.summary.FileWriter(
            self._hparams.log_dir + '/node_' + str(self.name))
        self._session = tf.compat.v1.Session(graph=self._graph)
        with self._graph.as_default():
            self._model_fn()
            self._session.run(tf.compat.v1.global_variables_initializer())

    def set_child(self, index, child):
        self._children[index] = child

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        logger.info('starting thread {}', self.name)
        self._thread.start()

    def stop(self):
        self._running = False
        logger.info('joining thread {}', self.name)
        self._thread.join()

    def spike(self, nounce, spikes, depth):
        if nounce in self._mem:
            return np.zeros((np.shape(spikes)[0], self._hparams.n_embedding))
        else:
            self._mem[nounce] = {'spikes': spikes}

        dspikes = self._get_dspikes(nounce, spikes, depth)
        self._mem[nounce]['dspikes'] = dspikes

        feeds = {self._spikes: spikes}
        for i, dspike in enumerate(dspikes):
            feeds[self._dspikes[i]] = dspike

        if (depth < hparams.max_depth):
            feeds[self._use_synthetic] = False
            embedding = self._session.run([self._embedding, self._syn_step],
                                          feeds)[0]
        else:
            feeds[self._use_synthetic] = True
            embedding = self._session.run(self._embedding, feeds)

        self._mem[nounce]['embedding'] = embedding
        return embedding

    def grade(self, nounce, spikes, grads, depth):
        if nounce not in self._mem:
            return

        # Computes the gradients for the local node as well as the gradients
        # for its children. Applies the gradients to the local node and sends the
        # children gradients downstream.
        feeds = {
            self._spikes: self._mem[nounce]['spikes'],
            self._egrads: grads,
            self._use_synthetic: False
        }
        dspikes = self._mem[nounce]['dspikes']
        for i, dspike in enumerate(dspikes):
            feeds[self._dspikes[i]] = dspike

        del self._mem[nounce]
        # Compute gradients for the children and apply the local step.
        dgrads = self._session.run(self._dgrads, feeds)

        if depth < self._hparams.max_depth:
            for i, child in enumerate(self._children):
                child.grade(nounce, spikes, dgrads[i][0], depth + 1)

    def _run(self):
        step = 0
        while self._running:
            batch_x, batch_y = self._mnist.train.next_batch(hparams.batch_size)
            self._train(batch_x, batch_y)
            if self._hparams.slow_step > 0:
                time.sleep(self._hparams.slow_step)
            if step % hparams.n_print == 0:
                self._summaries(step)
            step += 1

    def _summaries(self, step):
        batch_x, batch_y = self._mnist.train.next_batch(hparams.batch_size)
        train_acc = self._train(batch_x, batch_y)
        val_acc = self._test(batch_x, batch_y)
        summary = tf.Summary(value=[
            tf.Summary.Value(tag="accuracy", simple_value=val_acc),
        ])
        self._file_writer.add_summary(summary, step)
        self._file_writer.flush()
        zip_scores = [
            (child.name, "%.4f" % score)
            for child, score in list(zip(self._children, self._ema_deltaLij))
        ]
        logger.info('[{}] tr: {} val: {} fim: {}', self.name,
                    "%.4f" % train_acc, "%.4f" % val_acc, zip_scores)

    def _test(self, spikes, targets):
        # Run graph with synthetic inputs.
        dspikes = self._get_dspikes(None, spikes, 0, do_spike=False)
        feeds = {
            self._spikes: spikes,
            self._targets: targets,
            self._use_synthetic: True,
        }
        for i, dspike in enumerate(dspikes):
            feeds[self._dspikes[i]] = dspike
        return self._session.run(self._accuracy, feeds)

    def _train(self, spikes, targets):
        nounce = next_nounce()

        # Query children.
        dspikes = self._get_dspikes(nounce, spikes, 0)

        if self._hparams.trace:
            for i, dspk in enumerate(dspikes):
                logger.info("{} <- [{}{}]", self.name, self._children[i].name, dspk)

        feeds = {
            self._spikes: spikes,
            self._targets: targets,
            self._use_synthetic: False,
        }
        for i, dspike in enumerate(dspikes):
            feeds[self._dspikes[i]] = dspike

        # Compute the target loss from the input and make a training step.
        fetches = [
            self._tdgrads, self._accuracy, self._deltaLij, self._tstep,
            self._syn_loss, self._syn_step
        ]
        run_output = self._session.run(fetches, feeds)

        # EMA over FIM scores.
        deltaLij = run_output[2]
        for i, score in enumerate(deltaLij):
            prev_score = (self._ema_deltaLij[i] * (1 - self._hparams.score_ema))
            next_score = deltaLij[i] * (self._hparams.score_ema)
            self._ema_deltaLij[i] = prev_score + next_score

        # Recursively pass the gradients through the graph.
        train_grads = run_output[0]
        for i, child in enumerate(self._children):
            grads = train_grads[i][0]
            child.grade(nounce, spikes, grads, 0)

        # Return the batch accuracy.
        return run_output[1]

    def _get_dspikes(self, nounce, spikes, depth, do_spike=True):
        dspikes = []
        for child in self._children:
            if child and child.name != self.name and depth < self._hparams.max_depth and do_spike:
                if self._hparams.trace:
                    logger.info("{} -> o {}", self.name, child.name)
                dspikes.append(child.spike(nounce, spikes, depth + 1))
            else:
                dspikes.append(
                    np.zeros((np.shape(spikes)[0], self._hparams.n_embedding)))
                if self._hparams.trace:
                    logger.info("{} -> x {}", self.name, child.name)

        return dspikes

    def _model_fn(self):

        # Placeholders.
        # Spikes: inputs from the dataset of arbitrary batch_size.
        self._spikes = tf.compat.v1.placeholder(tf.float32,
                                                [None, self._hparams.n_inputs],
                                                's')
        # Dspikes: inputs from previous component. Size is the same as the embeddings produced
        # by this component.
        self._dspikes = []
        for _ in range(self._hparams.n_children):
            self._dspikes.append(
                tf.compat.v1.placeholder(tf.float32,
                                         [None, self._hparams.n_embedding],
                                         'd'))
        # Egrads: Gradient for this components embedding, passed by a parent.
        self._egrads = tf.compat.v1.placeholder(
            tf.float32, [None, self._hparams.n_embedding], 'g')
        # Targets: Supervised signals used during training and testing.
        self._targets = tf.compat.v1.placeholder(
            tf.float32, [None, self._hparams.n_targets], 't')
        # use_synthetic: Flag, use synthetic downstream spikes.
        self._use_synthetic = tf.compat.v1.placeholder(tf.bool,
                                                       shape=[],
                                                       name='use_synthetic')

        # Joiner weights and biases.
        jn_weights = {
            'jn_w1':
                tf.Variable(
                    tf.random.truncated_normal([
                        self._hparams.n_embedding * self._hparams.n_children,
                        self._hparams.n_jhidden1
                    ],
                                               stddev=0.01)),
            'jn_w2':
                tf.Variable(
                    tf.random.truncated_normal(
                        [self._hparams.n_jhidden1, self._hparams.n_jhidden2],
                        stddev=0.01)),
            'jn_w3':
                tf.Variable(
                    tf.random.truncated_normal(
                        [self._hparams.n_jhidden2, self._hparams.n_embedding],
                        stddev=0.01)),
        }
        jn_biases = {
            'jn_b1':
                tf.Variable(tf.constant(0.01, shape=[self._hparams.n_jhidden1])),
            'jn_b2':
                tf.Variable(tf.constant(0.01, shape=[self._hparams.n_jhidden2])),
            'jn_b3':
                tf.Variable(tf.constant(0.01,
                                        shape=[self._hparams.n_embedding])),
        }
        jn_vars = list(jn_weights.values()) + list(jn_biases.values())


        # Synthetic weights and biases.
        syn_weights = {
            'syn_w1':
                tf.Variable(
                    tf.random.truncated_normal(
                        [self._hparams.n_inputs, self._hparams.n_shidden1],
                        stddev=0.1)),
            'syn_w2':
                tf.Variable(
                    tf.random.truncated_normal(
                        [self._hparams.n_shidden1, self._hparams.n_shidden2],
                        stddev=0.1)),
            'syn_w3':
                tf.Variable(
                    tf.random.truncated_normal(
                        [self._hparams.n_shidden2, self._hparams.n_embedding],
                        stddev=0.1)),
        }
        syn_biases = {
            'syn_b1':
                tf.Variable(tf.constant(0.1, shape=[self._hparams.n_shidden1])),
            'syn_b2':
                tf.Variable(tf.constant(0.1, shape=[self._hparams.n_shidden2])),
            'syn_b3':
                tf.Variable(tf.constant(0.1,
                                        shape=[self._hparams.n_embedding])),
        }
        syn_vars = list(syn_weights.values()) + list(syn_biases.values())


        # Model weights and biases
        l_weights = {
            'w1':
                tf.Variable(
                    tf.random.truncated_normal([
                        self._hparams.n_inputs + self._hparams.n_embedding,
                        self._hparams.n_hidden1
                    ],
                                               stddev=0.1)),
            'w2':
                tf.Variable(
                    tf.random.truncated_normal(
                        [self._hparams.n_hidden1, self._hparams.n_hidden2],
                        stddev=0.1)),
            'w3':
                tf.Variable(
                    tf.random.truncated_normal(
                        [self._hparams.n_hidden2, self._hparams.n_embedding],
                        stddev=0.1)),
            'w4':
                tf.Variable(
                    tf.random.truncated_normal(
                        [self._hparams.n_embedding, self._hparams.n_targets],
                        stddev=0.1)),
        }
        l_biases = {
            'b1':
                tf.Variable(tf.constant(0.1, shape=[self._hparams.n_hidden1])),
            'b2':
                tf.Variable(tf.constant(0.1, shape=[self._hparams.n_hidden2])),
            'b3':
                tf.Variable(tf.constant(0.1,
                                        shape=[self._hparams.n_embedding])),
            'b4':
                tf.Variable(tf.constant(0.1, shape=[self._hparams.n_targets])),
        }
        l_vars = list(l_weights.values()) + list(l_biases.values())

        # Joiner network.
        if self._hparams.use_joiner_network:
            dspikes_concat = tf.concat(self._dspikes, axis=1)
            jn_hidden1 = tf.nn.relu(
                tf.add(tf.matmul(dspikes_concat, jn_weights['jn_w1']),
                       jn_biases['jn_b1']))
            jn_hidden2 = tf.nn.relu(
                tf.add(tf.matmul(jn_hidden1, jn_weights['jn_w2']),
                       jn_biases['jn_b2']))
            jn_embedding = tf.add(tf.matmul(jn_hidden2, jn_weights['jn_w3']),
                                  jn_biases['jn_b3'])
        else:
            jn_embedding = tf.add_n(self._dspikes)

        # Synthetic network.
        syn_hidden1 = tf.nn.relu(
            tf.add(tf.matmul(self._spikes, syn_weights['syn_w1']),
                   syn_biases['syn_b1']))
        syn_hidden2 = tf.nn.relu(
            tf.add(tf.matmul(syn_hidden1, syn_weights['syn_w2']),
                   syn_biases['syn_b2']))
        syn_embedding = tf.add(tf.matmul(syn_hidden2, syn_weights['syn_w3']),
                               syn_biases['syn_b3'])
        self._syn_loss = tf.reduce_mean(
            tf.nn.l2_loss(tf.stop_gradient(jn_embedding) - syn_embedding))

        # Switch between Synthetic embedding and Joiner embedding.
        input_embedding = tf.cond(
            tf.equal(self._use_synthetic, tf.constant(True)),
            true_fn=lambda: tf.stop_gradient(syn_embedding),
            false_fn=lambda: jn_embedding)

        # Local embedding network.
        input_layer = tf.concat([self._spikes, input_embedding], axis=1)
        hidden_layer1 = tf.nn.relu(
            tf.add(tf.matmul(input_layer, l_weights['w1']), l_biases['b1']))
        hidden_layer2 = tf.nn.relu(
            tf.add(tf.matmul(hidden_layer1, l_weights['w2']), l_biases['b2']))
        self._embedding = tf.nn.relu(
            tf.add(tf.matmul(hidden_layer2, l_weights['w3']), l_biases['b3']))

        # Target: softmax cross entropy over local network embeddings.
        logits = tf.add(tf.matmul(self._embedding, l_weights['w4']), l_biases['b4'])
        target_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._targets,
                                                       logits=logits))

        # Metrics: Calcuate accuracy.
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(self._targets, 1))
        self._accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        # Optimizer: The optimizer for this component.
        optimizer = tf.compat.v1.train.AdamOptimizer(
            self._hparams.learning_rate)

        # Synthetic network grads from synthetic loss.
        self._syn_grads = optimizer.compute_gradients(
            loss=self._syn_loss,
            var_list=syn_vars)

        # Downstream grads from target
        self._dgrads = optimizer.compute_gradients(loss=self._embedding,
                                                   var_list=self._dspikes,
                                                   grad_loss=self._egrads)

        # Local + joiner network grads from embedding grads.
        self._elgrads = optimizer.compute_gradients(
            loss=self._embedding,
            var_list=l_vars + jn_vars,
            grad_loss=self._egrads)

        # Downstream grads from target.
        self._tdgrads = optimizer.compute_gradients(loss=target_loss,
                                                    var_list=self._dspikes)

        # Local + joiner grads from target.
        self._tlgrads = optimizer.compute_gradients(
            loss=target_loss, var_list=l_vars + jn_vars)

        # Train step for synthetic inputs.
        self._syn_step = optimizer.apply_gradients(self._syn_grads)

        # Train step from embedding Local + joiner network grads.
        self._estep = optimizer.apply_gradients(self._elgrads)

        # Train step from target Local + joiner network grads.
        self._tstep = optimizer.apply_gradients(self._tlgrads)

        # FIM: Fishers information estimation.
        # Calculate contribution scores.
        # ∆Lij≈ ∑ gx·∆dj + 1/2N * ∆dj ∑ (gx∗gx)
        self._deltaLij = []
        for i, (gx, var) in enumerate(self._tdgrads):
            delta_d = -self._dspikes[i]
            g = tf.tensordot(delta_d, gx, axes=2)
            gxgx = tf.multiply(gx, gx)
            H = tf.tensordot(delta_d, gxgx, axes=2)
            score = tf.reduce_sum(g + H)
            #score = tf.Print(score, [i, score, delta_d])
            self._deltaLij.append(score)


def build_and_components(hparams):
    # Build async components.
    components = []
    for i in range(hparams.n_components):
        components.append(MACH(i, hparams))
    return components


def connect_components(hparams, components):
    for i in range(hparams.n_components):
        k = 0
        choice = random.sample(range(hparams.n_components), hparams.n_children)
        logger.info('node {}, choice:{}', i, choice)
        for el in choice:
            components[i].set_child(k, components[el])
            k += 1


def main(hparams):
    assert (hparams.n_components >= hparams.n_children + 1)

    components = build_and_components(hparams)
    connect_components(hparams, components)

    try:
        logger.info('Begin wait on main...')
        for component in components:
            if hparams.slow_step > 0:
                time.sleep(hparams.slow_step)
            component.start()
        while True:
            time.sleep(5)
    except:
        logger.debug('tear down.')
        for component in components:
            component.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch_size',
        default=50,
        type=int,
        help='The number of examples per batch. Default batch_size=128')
    parser.add_argument(
        '--learning_rate',
        default=1e-5,
        type=float,
        help='Component learning rate. Default learning_rate=1e-4')
    parser.add_argument(
        '--n_embedding',
        default=128,
        type=int,
        help='Size of embedding between components. Default n_embedding=128')
    parser.add_argument(
        '--n_children',
        default=2,
        type=int,
        help='The number of children in the graph. Default n_children=2')
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
    parser.add_argument('--n_hidden1',
                        default=512,
                        type=int,
                        help='Size of layer 1. Default n_hidden1=512')
    parser.add_argument('--n_hidden2',
                        default=512,
                        type=int,
                        help='Size of layer 1. Default n_hidden2=512')
    parser.add_argument(
        '--n_shidden1',
        default=512,
        type=int,
        help='Size of synthetic model hidden layer 1. Default n_shidden1=512')
    parser.add_argument(
        '--n_shidden2',
        default=512,
        type=int,
        help='Size of synthetic model hidden layer 2. Default n_shidden2=512')
    parser.add_argument(
        '--use_joiner_network',
        default=False,
        type=bool,
        help=
        'Do we combine downstream spikes using a trainable network. Default use_joiner_network=False'
    )
    parser.add_argument(
        '--n_jhidden1',
        default=512,
        type=int,
        help='Size of Joiner model hidden layer 1. Default n_shidden1=512')
    parser.add_argument(
        '--n_jhidden2',
        default=512,
        type=int,
        help='Size of Joinermodel hidden layer 2. Default n_shidden2=512')
    parser.add_argument(
        '--max_depth',
        default=1,
        type=int,
        help='Depth at which the synthetic inputs are used. Default max_depth=2'
    )
    parser.add_argument(
        '--n_print',
        default=100,
        type=int,
        help=
        'The number of iterations between print statements. Default n_print=100'
    )
    parser.add_argument('--slow_step',
                        default=-1,
                        type=float,
                        help='Slow down trainstep. Default slow_step=-1')
    parser.add_argument(
        '--score_ema',
        default=0.05,
        type=float,
        help='Moving average alpha for fishers scores. Default score_ema=0.05')
    parser.add_argument(
        '--log_dir',
        default='logs',
        type=str,
        help='location of tensorboard logs. Default log_dir=logs')
    parser.add_argument(
        '--trace',
        default=False,
        type=bool,
        help='Do trace. Default trace=false')

    hparams = parser.parse_args()

    main(hparams)
