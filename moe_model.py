"""Fully Asynchronous Learning Component.

This file contains the Mach class. This class runs on its own thread.
It contains a unqiue copy to the dataset and trains against the global loss.
During training it first queries the final activation layer ('embedding') from
its child and uses this as additional input to its own model. The class concurrently
trains a distillation model or ('synthetic input') using the child's outputs.
The target loss and the distilled model train concurrently.

During testing/validation or when a post-sequential node queries us, we do not
recursively query our own child, instead we use the distilled model as input to
our own network. This cuts the recursion and stops gradients from passing farther
through the network.

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

# Mixture of experts.
from utils_moe import noisy_top_k_gating
from utils_moe import SparseDispatcher

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
        self.children = None
        self.revenue = 0.0
        self.step = 0
        self.weights = {self.name: 1.0}

        self._mnist = mnist
        self._hparams = hparams
        self._tblogger = tblogger
        self._running = False
        self._graph = tf.Graph()
        self._session = tf.compat.v1.Session(graph=self._graph)

        with self._graph.as_default():
            self._model_fn()
            self._session.run(tf.compat.v1.global_variables_initializer())

    def set_children(self, children):
        """Set passed components as children.
        """
        assert(type(children) == list)
        assert(type(children[0]) == type(self))
        for child in children:
            self.weights[child.name] = 0.0
        self.children = children

    def start(self):
        """Start the training loop. Stops on call to stop.
        """
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        logger.info('starting thread {}', self.name)
        self._thread.start()

    def stop(self):
        """Joins training thread and stops.
        """
        self.running = False
        logger.info('joining thread {}', self.name)
        self._thread.join()

    def _run(self):
        """Loops train and test continually.
        """

        while self.running:
            # Training step.
            batch_x, batch_y = self._mnist.train.next_batch(self._hparams.batch_size)

            # Train child.
            self._run_graph(batch_x, batch_y, keep_prop=0.95, use_synthetic=False, do_train=True, do_metrics=False)

            # Train synthetic net.
            self._run_graph(batch_x, batch_y, keep_prop=0.95, use_synthetic=True, do_train=True, do_metrics=False)

            # Validation and tensorboard logs.
            if self.step % self._hparams.n_print == 0:

                # Train / Validation with and without synthetic models.
                val_x = self._mnist.test.images
                val_y = self._mnist.test.labels
                val_out = self._run_graph(val_x, val_y, keep_prop=1.0, use_synthetic=False, do_train=False, do_metrics=True)

                # Accuracy metrics.
                self._tblogger.log_scalar('validation', val_out['accuracy'][0], self.step)

                # Revenue metrics
                self._tblogger.log_scalar('revenue', val_out['revenue'][0], self.step)
                for i, child in enumerate(self.children):
                    self._tblogger.log_scalar('mask-' + str(child.name), val_out['masks'][i], self.step)
                for i, child in enumerate(self.children):
                    self._tblogger.log_scalar('weight-' + str(child.name), val_out['weights'][i+1], self.step)
                self._tblogger.log_scalar('inloop', val_out['weights'][0], self.step)

                # Set revenue and weights
                self.revenue = val_out['revenue'][0]
                for i, child in enumerate(self.children):
                    self.weights[child.name] = val_out['weights'][i+1]
                self.weights[self.name] = val_out['weights'][0]

                logger.info('{}: [acc {}, rev: {}, mask: {}, w: {}]', self.name, val_out['accuracy'][0], val_out['revenue'][0], [mask[0] for mask in val_out['masks']], val_out['weights'])
            if self.step > self._hparams.n_train_steps:
                self.running = False
            self.step+=1

    def _run_graph(self, spikes, targets, keep_prop, use_synthetic, do_train, do_metrics):
        """Runs the graph and returns fetch outputs.

        Args:
            spikes (numpy): mnist inputs [batch_size, 784]
            targets (numpy): mnist targets [batch_size, 10]
            keep_prop (float): dropout rate.
            use_synthetic (bool): do we use synthetic inputs or query child.
            do_train (bool): do we trigger training step.
        """

        # If not synthetic, use children.

        feeds = {
            self._spikes: spikes, # Mnist 784 input.
            self._targets: targets, # Mnist 1-hot Targets.
            self._keep_rate: keep_prop, # Dropout.
            self._use_synthetic: use_synthetic,
        }
        if not use_synthetic:
            # Run gating to get outgoing tensors.
            gate_feeds = {self._spikes: spikes}
            child_inputs = self._session.run(self._child_inputs, gate_feeds)
            for i, child in enumerate(self.children):
                feeds[self._child_outputs[i]] = child.spike(child_inputs[i])

        else:
            feeds[self._cspikes] = np.zeros((np.shape(spikes)[0], self._hparams.n_embedding))

        # Build fetches.
        fetches = {}

        if do_train:
            fetches['target_step'] = self._tstep
            fetches['gate_step'] = self._gstep

        # We train the synthetic model when we query our child.
        if not use_synthetic:
            fetches['synthetic_loss'] = self._syn_loss # Distillation loss.

        if do_train:
            fetches['synthetic_step'] = self._syn_step # Synthetic step.
            fetches['child_gradients'] = self._tdgrads

        if do_metrics:
            fetches['accuracy'] = self._accuracy, # Classification accuracy.
            fetches['target_loss'] = self._target_loss, # Target accuracy.
            fetches['load'] = self._load
            fetches['revenue'] = self._revenue
            fetches['masks'] = self._masks
            fetches['weights'] = self._weights

        # Run graph.
        run_output = self._session.run(fetches, feeds)

        # Pass the gradients to the child.
        if do_train and not use_synthetic:
            for i, child in enumerate(self.children):
                child.grade(child_inputs[i], run_output['child_gradients'][i][0])

        # Return the batch accuracy.
        return run_output

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
        self._spikes = tf.compat.v1.placeholder(tf.float32, [None, self._hparams.n_inputs], name='spikes')

        # Egrads: Gradient for this component's embedding, passed by a parent.
        self._egrads = tf.compat.v1.placeholder(tf.float32, [None, self._hparams.n_embedding], name='embedding_grads')

        # Targets: Supervised signals used during training and testing.
        self._targets = tf.compat.v1.placeholder(tf.float32, [None, self._hparams.n_targets], name='targets')

        # use_synthetic: Flag, use synthetic downstream spikes.
        self._use_synthetic = tf.compat.v1.placeholder(tf.bool, shape=[], name='use_synthetic')

        # dropout prob.
        self._keep_rate = tf.placeholder_with_default(1.0, shape=(), name='keep_rate')

        # Gating Network
        with tf.compat.v1.variable_scope("gate"):
            gates, self._load = noisy_top_k_gating(   self._spikes,
                                                      self._hparams.n_components - 1,
                                                      train=True )
            dispatcher = SparseDispatcher(self._hparams.n_components-1, gates)
            self._child_inputs = dispatcher.dispatch(self._spikes)

            importance = tf.linalg.normalize(tf.reduce_sum(gates, 0))[0]
            inloop = tf.Variable(tf.constant([1.0]))
            self._weights = tf.linalg.normalize(tf.concat([inloop, importance], axis=0))[0]
            self._revenue = tf.slice(self._weights, [0], [1])

            # Join child inputs.
            self._child_outputs = []
            self._masks = []
            for i in range(self._hparams.n_components - 1):
                child_output = tf.compat.v1.placeholder_with_default(tf.zeros([tf.shape(self._child_inputs[i])[0], self._hparams.n_embedding]), [None, self._hparams.n_embedding], name='einput' + str(i))

                # Apply mask to the output.
                child_mask = tf.nn.relu(tf.slice(self._weights, [i], [1]) - 0.2)
                masked_child_output = child_mask * child_output

                #self._child_outputs.append(masked_child_output)
                self._masks.append(child_mask)
                self._child_outputs.append(masked_child_output)

            # Child spikes if needed.
            self._cspikes = dispatcher.combine(self._child_outputs)


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
        synthetic_network_variables = list(syn_weights.values()) + list(syn_biases.values())

        # Weights and biases
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
        local_network_variables = list(weights.values()) + list(biases.values())

        # Syn_embedding: The synthetic input, produced by distilling the child component with a local model.
        syn_hidden1 = tf.nn.relu(tf.add(tf.matmul(self._spikes, syn_weights['syn_w1']), syn_biases['syn_b1']))
        syn_hidden2 = tf.nn.relu(tf.add(tf.matmul(syn_hidden1, syn_weights['syn_w2']), syn_biases['syn_b2']))
        syn_cspikes = tf.add(tf.matmul(syn_hidden2, syn_weights['syn_w3']), syn_biases['syn_b3'])
        self._syn_loss = tf.reduce_mean(tf.nn.l2_loss(tf.stop_gradient(self._cspikes) - syn_cspikes))
        tf.compat.v1.summary.scalar("syn_loss", self._syn_loss)

        # Switch between synthetic embedding or true_embedding
        self._downstream = tf.cond(tf.equal(self._use_synthetic, tf.constant(True)),
                              true_fn=lambda: syn_cspikes,
                              false_fn=lambda: self._cspikes)

        # Embedding: the embedding passes to the parent.
        self._input_layer = tf.concat([self._spikes, self._downstream], axis=1)
        hidden_layer1 = tf.nn.relu(tf.add(tf.matmul(self._input_layer, weights['w1']), biases['b1']))
        hidden_layer2 = tf.nn.relu(tf.add(tf.matmul(hidden_layer1, weights['w2']), biases['b2']))
        drop_hidden_layer2 = tf.nn.dropout(hidden_layer2, self._keep_rate)
        self._embedding = tf.reshape(tf.nn.relu(tf.add(tf.matmul(drop_hidden_layer2, weights['w3']), biases['b3'])), [-1, self._hparams.n_embedding])


        # Target: the mnist target.
        self._logits = tf.add(tf.matmul(self._embedding, weights['w4']), biases['b4'])
        self._target_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._targets, logits=self._logits))

        self._full_loss = self._target_loss - self._revenue

        # Optimizer: The optimizer for this component, could be different accross components.
        optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)

        # syn_grads: Gradient terms for the synthetic inputs.
        self._syn_grads = optimizer.compute_gradients(loss=self._syn_loss + self._full_loss, var_list=synthetic_network_variables)

        # Embedding grads: Here, we compute the gradient terms for the embedding with respect
        # to the gradients passed from the parent (a.k.a egrads). Dgrads is the gradient for
        # the downstream component (child) and elgrads are the gradient terms for the the local
        # FFNN.
        self._cgrads = optimizer.compute_gradients(loss=self._embedding, var_list=self._child_outputs, grad_loss=self._egrads)[0][0]
        self._elgrads = optimizer.compute_gradients(loss=self._embedding, var_list=local_network_variables, grad_loss=self._egrads)

        # Gradients from target: Here, we compute the gradient terms for the downstream child and
        # the local variables but with respect to the target loss. These get sent downstream and used to
        # optimize the local variables.
        self._tdgrads = optimizer.compute_gradients(loss=self._full_loss, var_list=self._child_outputs)
        self._tlgrads = optimizer.compute_gradients(loss=self._full_loss, var_list=local_network_variables)
        self._tggrads = optimizer.compute_gradients(loss=self._full_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="gate"))

        # Syn step: Train step which applies the synthetic input grads to the synthetic input model.
        self._syn_step = optimizer.apply_gradients(self._syn_grads)

        # Embedding trainstep: Train step which applies the gradients calculated w.r.t the gradients
        # from a parent.
        self._estep = optimizer.apply_gradients(self._elgrads)

        # Target trainstep: Train step which applies the gradients calculated w.r.t the target loss.
        self._tstep = optimizer.apply_gradients(self._tlgrads)

        # Gate step.
        self._gstep = optimizer.apply_gradients(self._tggrads)

        # Metrics:

        # Accuracy
        correct = tf.equal(tf.argmax(self._logits, 1), tf.argmax(self._targets, 1))
        self._accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
