"""Fully Asynchronous Learning Component.

This file contains the Mach class.

Example:
        $ mach = Mach ( name = 0,
                        mnist = load_data_and_constants(),
                        hparams = hparams_from_args(),
                        tblogger = TBLogger())

"""
from loguru import logger
import numpy as np
import tensorflow as tf
import threading

class Mach:
    def __init__(self, name, mnist, hparams, tblogger):
        """Initialize a Mach learning component.
        Args:
            name: component name.
            mnist: mnist dataset from utils.load_data_and_constants
            hparams: component hyperparameters from arguments.
            tblogger: tensorboard logger class.
        """
        self.name = name
        self._mnist = mnist
        self._hparams = hparams
        self._tblogger = tblogger
        self._child = None
        self._running = False
        self._graph = tf.Graph()
        self._session = tf.compat.v1.Session(graph=self._graph)
        with self._graph.as_default():
            self._model_fn()
            self._session.run(tf.compat.v1.global_variables_initializer())

    def set_child(self, child):
        """Sets passed component as child.
        """
        assert(type(child) == type(self))
        self._child = child

    def start(self):
        """Start the training loop. Stops on call to stop.
        """
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        logger.info('starting thread {}', self.name)
        self._thread.start()

    def stop(self):
        """Joins trainign thread stops training.
        """
        self.running = False
        logger.info('joining thread {}', self.name)
        self._thread.join()

    def _run(self):
        """Loops train and test continually.
        """
        step = 0
        while self.running:
            # Run train step.
            batch_x, batch_y = self._mnist.train.next_batch(self._hparams.batch_size)
            self._train(batch_x, batch_y)

            # Run test step and print.
            if step % self._hparams.n_print == 0:
                train_acc = self._train(batch_x, batch_y)
                val_acc = self._test(self._mnist.test.images, self._mnist.test.labels)
                self._tblogger.log_scalar('val_accuracy', val_acc, step)
                self._tblogger.log_scalar('tr_accuracy', train_acc, step)
                logger.info('{}: train {}, validation {}', self.name, train_acc, val_acc)
            if step > self._hparams.n_train_steps:
                self.running = False
            step+=1

    def _test(self, spikes, targets):
        """Runs the test graph using synthetic inputs.
        Args:
            spikes (numpy): mnist inputs [batch_size, 784]
            targets (numpy): mnist targets [batch_size, 10]
        """
        cspikes = np.zeros((np.shape(spikes)[0], self._hparams.n_embedding))
        feeds = {
            self._spikes: spikes,
            self._cspikes: cspikes,
            self._targets: targets,
            self._use_synthetic: True,
            self._keep_rate: 1.0,
        }
        # Return the model accuracy.
        return self._session.run(self._accuracy, feeds)

    def _train(self, spikes, targets):
        """Runs the traininng graph.

        First queries child to retrieve inputs to the local graph. Then computes
        the local target loss. Gradients apply to the local model and are passed
        downstream to the children.

        Args:
            spikes (numpy): mnist inputs [batch_size, 784]
            targets (numpy): mnist targets [batch_size, 10]
        """

        # Query child if exists.
        cspikes =  np.zeros((np.shape(spikes)[0], self._hparams.n_embedding))
        if self._child:
            cspikes = self._child.spike(spikes)

        # Build feeds.
        feeds = {
            self._spikes: spikes, # Mnist 784 input.
            self._cspikes: cspikes, # Child inputs.
            self._targets: targets, # Mnist 1-hot Targets.
            self._use_synthetic: False, # Do not use synthetic inputs.
            self._keep_rate: 0.96, # Dropout.
        }

        # Build fetches.
        fetches = {
            'target_child_graients': self._tdgrads,
            'accuracy': self._accuracy,
            'target_step': self._tstep,
            'synthetic_loss': self._syn_loss,
            'synthetic_step': self._syn_step
        }

        # Run graph.
        run_output = self._session.run(fetches, feeds)

        # Pass the gradients to the child.
        if self._child:
            self._child.grade(spikes, run_output['target_child_graients'])

        # Return the batch accuracy.
        return run_output['accuracy']

    def spike(self, spikes):
        """ External query on this node.

        Spikes the local node returning its representational output given the
        input.

        Args:
            spikes (numpy): mnist inputs [batch_size, 784]
        """

        # Return using synthetic inputs as input to this component.
        zeros = np.zeros((np.shape(spikes)[0], self._hparams.n_embedding))
        feeds = {
            self._spikes: spikes, # Mnist inputs.
            self._cspikes: zeros, # Zeros from children (not used)
            self._use_synthetic: True, # Use synthetic children.
            self._keep_rate: 1.0, # No dropout.
        }
        # Return the embedding.
        return self._session.run(self._embedding, feeds)

    def grade(self, spikes, grads):
        """ Grade the child node.

        Computes and applies the gradients to the local node given the
        passed signal.

        Args:
            spikes (numpy): mnist inputs [batch_size, 784]
            grads (numpy): gradients [batch_size, n_embedding]
        """

        zeros = np.zeros((np.shape(spikes)[0], self._hparams.n_embedding))
        feeds = {
            self._spikes: spikes, # Spikes from query.
            self._egrads: grads, # Embedding gradients from parent.
            self._cspikes: zeros, # Zeros from children.
            self._use_synthetic: False, # Do not use Synthetic.
            self._keep_rate: 1.0 # No Dropout.
        }
        # Run the embedding step.
        self._session.run([self._estep], feeds)

    def _model_fn(self):
        """ Tensorflow model function

        Builds the model: See (https://www.overleaf.com/read/fvyqcmybsgfj)

        """

        # Placeholders.
        # Spikes: inputs from the dataset of arbitrary batch_size.
        self._spikes = tf.compat.v1.placeholder(tf.float32, [None, self._hparams.n_inputs])
        # Cspikes: inputs from previous component. Size is the same as the embeddings produced
        # by this component.
        self._cspikes = tf.compat.v1.placeholder(tf.float32, [None, self._hparams.n_embedding])
        # Egrads: Gradient for this components embedding, passed by a parent.
        self._egrads = tf.compat.v1.placeholder(tf.float32, [None, self._hparams.n_embedding])
        # Targets: Supervised signals used during training and testing.
        self._targets = tf.compat.v1.placeholder(tf.float32, [None, self._hparams.n_targets])
        # use_synthetic: Flag, use synthetic downstream spikes.
        self._use_synthetic = tf.compat.v1.placeholder(tf.bool, shape=[], name='use_synthetic')
        # dropout prob
        self._keep_rate = tf.placeholder_with_default(1.0, shape=())

        # Synthetic weights and biases.
        syn_weights = {
            'syn_w1': tf.Variable(tf.random.truncated_normal([self._hparams.n_inputs , self._hparams.n_shidden1], stddev=0.1)),
            'syn_w2': tf.Variable(tf.random.truncated_normal([self._hparams.n_shidden1, self._hparams.n_shidden2], stddev=0.1)),
            'syn_w3': tf.Variable(tf.random.truncated_normal([self._hparams.n_shidden2, self._hparams.n_embedding], stddev=0.1)),
        }
        syn_biases = {
            'syn_b1': tf.Variable(tf.constant(0.1, shape=[self._hparams.n_shidden1])),
            'syn_b2': tf.Variable(tf.constant(0.1, shape=[self._hparams.n_shidden2])),
            'syn_b3': tf.Variable(tf.constant(0.1, shape=[self._hparams.n_embedding])),
        }

        # Weights and biases + Synthetic weights and biases.
        # Each component has one hidden layer. Projection is from the input dimension, concatenated
        # with the embeddings from the previous component.
        weights = {
            'w1': tf.Variable(tf.random.truncated_normal([self._hparams.n_inputs + self._hparams.n_embedding, self._hparams.n_hidden1], stddev=0.1)),
            'w2': tf.Variable(tf.random.truncated_normal([self._hparams.n_hidden1, self._hparams.n_hidden2], stddev=0.1)),
            'w3': tf.Variable(tf.random.truncated_normal([self._hparams.n_hidden2, self._hparams.n_embedding], stddev=0.1)),
            'w4': tf.Variable(tf.random.truncated_normal([self._hparams.n_embedding, self._hparams.n_targets], stddev=0.1)),
            }
        biases = {
            'b1': tf.Variable(tf.constant(0.1, shape=[self._hparams.n_hidden1])),
            'b2': tf.Variable(tf.constant(0.1, shape=[self._hparams.n_hidden2])),
            'b3': tf.Variable(tf.constant(0.1, shape=[self._hparams.n_embedding])),
            'b4': tf.Variable(tf.constant(0.1, shape=[self._hparams.n_targets])),
        }

        # Syn_embedding: The synthetic input, produced by distilling the child component with a local model.
        syn_hidden1 = tf.nn.relu(tf.add(tf.matmul(self._spikes, syn_weights['syn_w1']), syn_biases['syn_b1']))
        syn_hidden2 = tf.nn.relu(tf.add(tf.matmul(syn_hidden1, syn_weights['syn_w2']), syn_biases['syn_b2']))
        syn_cspikes = tf.add(tf.matmul(syn_hidden2, syn_weights['syn_w3']), syn_biases['syn_b3'])
        #syn_cspikes = tf.Print(syn_cspikes, [self._cspikes, syn_cspikes], summarize=100000)
        self._syn_loss = tf.reduce_mean(tf.nn.l2_loss(tf.stop_gradient(self._cspikes) - syn_cspikes))
        tf.compat.v1.summary.scalar("syn_loss", self._syn_loss)

        # Switch between synthetic embedding or true_embedding
        cspikes = tf.cond(tf.equal(self._use_synthetic, tf.constant(True)),
                              true_fn=lambda: tf.stop_gradient(syn_cspikes),
                              false_fn=lambda: self._cspikes)

        # Embedding: Apply the hidden layer to the spikes and embeddings from the previous component.
        # The embedding is the output for this component passed to its parents.
        input_layer = tf.concat([self._spikes, cspikes], axis=1)
        hidden_layer1 = tf.nn.relu(tf.add(tf.matmul(input_layer, weights['w1']), biases['b1']))
        hidden_layer2 = tf.nn.relu(tf.add(tf.matmul(hidden_layer1, weights['w2']), biases['b2']))
        drop_hidden_layer2 = tf.nn.dropout(hidden_layer2, self._keep_rate)
        self._embedding = tf.nn.relu(tf.add(tf.matmul(drop_hidden_layer2, weights['w3']), biases['b3']))


        # Target: Apply a softmax over the embeddings. This is the loss from the local network.
        # The loss on the target and the loss from the parent averaged.
        logits = tf.add(tf.matmul(self._embedding, weights['w4']), biases['b4'])
        target_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._targets, logits=logits))

        # Metrics: We calcuate accuracy here because we are working with MNIST.
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(self._targets, 1))
        self._accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        # Optimizer: The optimizer for this component, could be different accross components.
        optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)

        # syn_grads: Here, we compute the gradient terms for the synthetic inputs.
        self._syn_grads = optimizer.compute_gradients(loss=self._syn_loss, var_list=list(syn_weights.values()) + list(syn_biases.values()))

        # Embedding grads: Here, we compute the gradient terms for the embedding with respect
        # to the gradients passed from the parent (a.k.a egrads). Dgrads is the gradient for
        # the downstream component (child) and elgrads are the gradient terms for the the local
        # FFNN.
        self._cgrads = optimizer.compute_gradients(loss=self._embedding, var_list=[self._cspikes], grad_loss=self._egrads)[0][0]
        self._elgrads = optimizer.compute_gradients(loss=self._embedding, var_list=tf.compat.v1.trainable_variables(), grad_loss=self._egrads)

        # Gradients from target: Here, we compute the gradient terms for the downstream child and
        # the local variables but with respect to the target loss. These get sent downstream and used to
        # optimize the local variables.
        self._tdgrads = optimizer.compute_gradients(loss=target_loss, var_list=[self._cspikes])[0][0]
        self._tlgrads = optimizer.compute_gradients(loss=target_loss, var_list=tf.compat.v1.trainable_variables())

        # Syn step: Train step which applies the synthetic input grads to the synthetic input model.
        self._syn_step = optimizer.apply_gradients(self._syn_grads)

        # Embedding trainstep: Train step which applies the gradients calculated w.r.t the gradients
        # from a parent.
        self._estep = optimizer.apply_gradients(self._elgrads)

        # Target trainstep: Train step which applies the gradients calculated w.r.t the target loss.
        self._tstep = optimizer.apply_gradients(self._tlgrads)
