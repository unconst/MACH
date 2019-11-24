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
import queue

def load_data_and_constants(hparams):
    '''Returns the dataset and sets hparams.n_inputs and hparamsn_targets.'''
    # Load mnist data
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    hparams.n_inputs = 784
    hparams.n_targets = 10
    return mnist, hparams

def next_nounce():
    # Random number from large range.
    return random.randint(0, 1000000000)


class MACH:

    def __init__(self, name, hparams):
        # Mach name.
        self.name = name
        # Dataset.
        self._mnist, self._hparams = load_data_and_constants(hparams)
        # Children set.
        self._children = [None for _ in range(self._hparams.n_children)]
        # Queue of local network gradients.
        self._grad_queue = queue.LifoQueue(maxsize=-1)
        # TF graph.
        self._graph = tf.Graph()
        self._session = tf.compat.v1.Session(graph=self._graph)
        # Build local graph and init.
        with self._graph.as_default():
            self._model_fn()
            self._session.run(tf.compat.v1.global_variables_initializer())

    def set_child(self, index, child):
        self._children[index] = child

    def start(self):
        """ Starts run thread
        """
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        logger.info('starting thread {}', self.name)
        self._thread.start()

    def stop(self):
        """ Stops run thread
        """
        self._running = False
        logger.info('joining thread {}', self.name)
        self._thread.join()

    def _run(self):
        """ Runs train and test loop.
        """
        step = 0
        acc = 0
        while self._running:
            batch_x, batch_y = self._mnist.train.next_batch(hparams.batch_size)
            acc = acc * (0.95) + self._train(batch_x, batch_y) * 0.05
            if step % hparams.local_step == 0:
                self._local_step()
            if step % hparams.n_print == 0:
                logger.info('[{}] tr acc {}', self.name, acc)
            step += 1

    def _test(self, spikes, targets):
        """ Tests the local graph using synthetic inputs.
        Args:
            spikes: (numpy) graph inputs [batch_size, n_inputs]
            targets: (numpy) supervised target [batch_size, n_targets]
        """
        # Run graph with synthetic inputs.
        feeds = {
            self._spikes: spikes,
            self._targets: targets,
            self._use_synthetic: True,
        }
        for i, child in enumerate(self._children):
            feeds[self._cspikes[i]] = np.zeros((self._hparams.batch_size, self._hparams.n_embedding))
        # Return testing accuracy.
        return self._session.run(self._accuracy, feeds)

    def _normalize_grads(self, grads):
        """ Averages gradients network gradients.
        Args:
            grads: [ el: ( name: int, grad: [(grad, var)] )  ] local gradients to normalize
        """
        # Build list of zero gradients.
        grad_sum = []
        for g in grads[0][1]:
            grad_sum.append(np.zeros(g[0].shape))

        # Sum gradients accross list.
        for grad in grads:
            name = grad[0]
            gradients = grad[1][0]
            for i, grad in enumerate(gradients):
                grad_sum[i] += grad[0]

        # Average each gradient.
        for i, _ in enumerate(grad_sum):
            grad_sum[i] = grad_sum[i]/len(grads)

        # Return
        return grad_sum

    def _local_step(self):
        """ Applies the pending local network gradients.
        """
        if self._grad_queue.qsize() == 0:
            return

        # Empty the queue.
        local_grads = []
        while not self._grad_queue.empty():
            local_grads.append(self._grad_queue.get())

        # Normalize gradients into a single batch.
        norm_grads = self._normalize_grads(local_grads)

        # Apply the step.
        feeds = {}
        for i, grad in enumerate(norm_grads):
            feeds[self._l_pgrads[i][0]] = grad

        # Run local network step.
        fetches = {
            'l_pstep': self._l_pstep,
        }
        self._session.run(fetches, feeds)


    def _train(self, spikes, targets):
        """ trains the target joiner and synthetic graph, produces gradients
        for local network and sends gradients to children.
        Args:
            spikes: (numpy) graph inputs [batch_size, n_inputs]
            targets: (numpy) supervised target [batch_size, n_targets]

        Returns:
            accuracy: target accuracy.
        """
        # Build feeds.
        feeds = {
            self._spikes: spikes,
            self._targets: targets,
            self._use_synthetic: False # Use Joiner with child spikes.
        }
        # Fill child spikes by calling children.
        for i, child in enumerate(self._children):
            feeds[self._cspikes[i]] = child.spike(spikes)

        # Build fetches
        fetches = {
            'c_tgrads': self._c_tgrads, # gradients for children.
            'l_tgrads': self._l_tgrads, # gradients for local network.
            'accuracy': self._accuracy, # supervised accuracy.
            't_step': self._t_step, # train step for target network.
            'jn_tstep': self._jn_tstep, # train step for joiner network.
            'syn_step': self._syn_step # train step for synthetic network.
        }

        # Run graph.
        run_output = self._session.run(fetches, feeds)

        # Push local network gradients for later.
        self._grad_queue.put((self.name, run_output['l_tgrads']))

        # Send gradients to children.
        for i, child in enumerate(self._children):
            child.grade(self.name, spikes, run_output['c_tgrads'][i][0])

        # Return the batch accuracy.
        return run_output['accuracy']

    def spike(self, spikes):
        """ spikes: forward pass through graph using synthetic inputs.
        Args:
            spikes: (numpy) graph inputs [batch_size, n_inputs]
        Returns:
            embedding: (numpy) local network embedding [batch_size, n_embedding]
        """
        # Build feeds.
        feeds = {
            self._spikes: spikes,
            self._use_synthetic: True,
        }

        # Fill child spikes with dummy inputs.
        for i, child in enumerate(self._children):
            feeds[self._cspikes[i]] = np.zeros((self._hparams.batch_size, self._hparams.n_embedding))

        # Fetch local network embedding
        fetches = {
            'embedding': self._embedding,
        }

        # Run graph.
        run_output = self._session.run(fetches, feeds)

        # Retuen local embedding
        return run_output['embedding']

    def grade(self, name, spikes, pgrads):
        """ grade: backward passes gradients from parent
        Args:
            spikes: (numpy) graph inputs [batch_size, n_inputs]
            pgrades: (numpy) parent gradients [batch_size, n_embedding]
        Returns:
            None
        """
        # Build feeds passing inputs gradients and setting synthetic to True.
        feeds = {
            self._spikes: spikes,
            self._pgrads: pgrads,
            self._use_synthetic: True
        }

        # Fill child spikes with dummy zeros.
        for i, child in enumerate(self._children):
            feeds[self._cspikes[i]] = np.zeros((self._hparams.batch_size, self._hparams.n_embedding))

        # Fetch gradients for the local network only.
        fetches = {
            'l_pgrads': self._l_pgrads
        }

        # Run fetch.
        run_output = self._session.run(fetches, feeds)

        # Append the gradients to the grad queue for later.
        self._grad_queue.put((name, run_output['l_pgrads']))


    def _model_fn(self):
        """ _model_fn: build the entire graph.
        Args:
            None
        Returns:
            None
        """

        # Placeholders:

        # Spikes: float32 [batch_size, n_inputs]
        self._spikes = tf.compat.v1.placeholder(tf.float32,
                                                [None, self._hparams.n_inputs],
                                                's')

        # Targets: float32 [batch_size, n_targets]
        self._targets = tf.compat.v1.placeholder(
            tf.float32, [None, self._hparams.n_targets], 't')


        # Parent gradients: float32 [batch_size, n_embedding]
        self._pgrads = tf.compat.v1.placeholder(
            tf.float32, [None, self._hparams.n_embedding], 'g')

        # Synthetic network switch: bool []
        self._use_synthetic = tf.compat.v1.placeholder(tf.bool,
                                                       shape=[],
                                                       name='use_synthetic')

        # Child Spikes:  (k-1) * [None, n_embedding]
        self._cspikes = []
        for _ in self._children:
            self._cspikes.append(
                tf.compat.v1.placeholder(tf.float32,
                                         [None, self._hparams.n_embedding],
                                         'd'))


        # Variables:

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
                tf.Variable(tf.constant(0.01,
                                        shape=[self._hparams.n_jhidden1])),
            'jn_b2':
                tf.Variable(tf.constant(0.01,
                                        shape=[self._hparams.n_jhidden2])),
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


        # Local weights and biases
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
                        stddev=0.1))
        }
        l_biases = {
            'b1':
                tf.Variable(tf.constant(0.1, shape=[self._hparams.n_hidden1])),
            'b2':
                tf.Variable(tf.constant(0.1, shape=[self._hparams.n_hidden2])),
            'b3':
                tf.Variable(tf.constant(0.1,
                                        shape=[self._hparams.n_embedding]))
        }
        l_vars = list(l_weights.values()) + list(l_biases.values())


        # Target weights and biases.
        t_weights = {
            'w1':
                tf.Variable(
                    tf.random.truncated_normal(
                        [self._hparams.n_embedding, self._hparams.n_targets],
                        stddev=0.1))
        }
        t_biases = {
            'b1': tf.Variable(tf.constant(0.1, shape=[self._hparams.n_targets]))
        }
        t_vars = list(t_weights.values()) + list(t_biases.values())


        # Networks:

        # Joiner network. [-1, n_embedding * (k-1)] --> [-1, n_embedding]
        dspikes_concat = tf.concat(self._cspikes, axis=1)
        jn_hidden1 = tf.nn.relu(
            tf.add(tf.matmul(dspikes_concat, jn_weights['jn_w1']),
                   jn_biases['jn_b1']))
        jn_hidden2 = tf.nn.relu(
            tf.add(tf.matmul(jn_hidden1, jn_weights['jn_w2']),
                   jn_biases['jn_b2']))
        jn_embedding = tf.add(tf.matmul(jn_hidden2, jn_weights['jn_w3']),
                              jn_biases['jn_b3'])


        # Synthetic network. [-1, n_inputs] --> [-1, n_embedding]
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



        # Switch: Synthetic vs Joiner: [-1, n_embedding]
        input_embedding = tf.cond(
            tf.equal(self._use_synthetic, tf.constant(True)),
            true_fn=lambda: tf.stop_gradient(syn_embedding),
            false_fn=lambda: jn_embedding)



        # Local network. [-1, n_embedding] --> [-1, n_embedding]
        input_layer = tf.concat([self._spikes, input_embedding], axis=1)
        hidden_layer1 = tf.nn.relu(
            tf.add(tf.matmul(input_layer, l_weights['w1']), l_biases['b1']))
        hidden_layer2 = tf.nn.relu(
            tf.add(tf.matmul(hidden_layer1, l_weights['w2']), l_biases['b2']))
        self._embedding = tf.nn.relu(
            tf.add(tf.matmul(hidden_layer2, l_weights['w3']), l_biases['b3']))



        # Target network. [-1, n_embedding] --> [-1, n_target]
        logits = tf.add(tf.matmul(self._embedding, t_weights['w1']),
                        t_biases['b1'])
        target_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._targets,
                                                       logits=logits))


        # Gradients:

        # Optimizer for target joiner and synthetic networks.
        optimizer = tf.compat.v1.train.AdamOptimizer(
            self._hparams.learning_rate)

        # Optimizer for local network.
        l_optimizer = tf.compat.v1.train.AdamOptimizer(
            self._hparams.l_learning_rate)

        # Synthetic network grads. (from synthetic loss)
        self._syn_grads = optimizer.compute_gradients(  loss=self._syn_loss,
                                                        var_list=syn_vars)


        # Child grads. (from target)
        self._c_tgrads = optimizer.compute_gradients(   loss=target_loss,
                                                        var_list=self._cspikes)

        # Child grads. (from parent)
        self._c_pgrads = optimizer.compute_gradients(   loss=self._embedding,
                                                        var_list=self._cspikes,
                                                        grad_loss=self._pgrads)

        # Joiner network grads. (from target)
        self._jn_tgrads = optimizer.compute_gradients(  loss=target_loss,
                                                        var_list=jn_vars)

        # Joiner network grads. (from parent)
        self._jn_pgrads = optimizer.compute_gradients(  loss=self._embedding,
                                                        var_list=jn_vars,
                                                        grad_loss=self._pgrads)

        # Local network grads. (from target)
        self._l_tgrads = optimizer.compute_gradients(   loss=target_loss,
                                                        var_list=l_vars)

        # Local network grads. (from parent)
        self._l_pgrads = optimizer.compute_gradients(   loss=self._embedding,
                                                        var_list=l_vars,
                                                        grad_loss=self._pgrads)

        # Target network grads (from target)
        self._t_grads = optimizer.compute_gradients(loss=target_loss,
                                                    var_list=t_vars)


        # Train steps:

        # Synthetic step (from synthetic).
        self._syn_step = optimizer.apply_gradients(self._syn_grads)

        # Joiner step (from target).
        self._jn_tstep = optimizer.apply_gradients(self._jn_tgrads)

        # Joiner step (from parent).
        self._jn_pstep = optimizer.apply_gradients(self._jn_pgrads)

        # Local step (from target).
        self._l_tstep = optimizer.apply_gradients(self._l_tgrads)

        # Local step (from parent).
        self._l_pstep = l_optimizer.apply_gradients(self._l_pgrads)

        # Target step (from target).
        self._t_step = optimizer.apply_gradients(self._t_grads)


        # Metrics:

        # accuracy.
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(self._targets, 1))
        self._accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))



def build_and_components(hparams):
    # Build async components.
    components = []
    for i in range(hparams.n_components):
        components.append(MACH(i, hparams))
    return components


def connect_components(hparams, components):
    for i in range(hparams.n_components):
        k = 0
        for j in range(hparams.n_components):
            if i != j:
                components[i].set_child(k, components[j])
                k += 1


def main(hparams):
    logger.info('hparams: {}', hparams)

    # Build k graph.
    hparams.n_children = hparams.n_components - 1
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
        '--local_step',
        default=1,
        type=int,
        help='When to apply the local step. Default local_step=50')
    parser.add_argument(
        '--learning_rate',
        default=1e-4,
        type=float,
        help='Target learning rate, applied to the logit softmax. Default learning_rate=1e-4')
    parser.add_argument(
        '--l_learning_rate',
        default=1e-5,
        type=float,
        help='Local model learning rate, Applied to internal embedding. Default learning_rate=1e-4')
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
        '--n_print',
        default=10,
        type=int,
        help=
        'The number of iterations between print statements. Default n_print=100'
    )
    parser.add_argument(
        '--slow_step',
        default=-1,
        type=int,
        help=
        'Run slow training steps for debugging. Number of seconds between steps. Default --slow_step=-1'
    )

    hparams = parser.parse_args()

    main(hparams)
