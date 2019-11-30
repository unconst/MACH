"""Fully Asynchronous Learning Component.

This file contains the Mach class. This class runs on its own thread.
It contains a unqiue copy to the dataset and trains against the global loss.
During training it first queries the final activation layer ('embedding') from
its child and uses this as additional input to its own model. The class concurrently
trains a distillation model or ('synthetic input') using the child's outputs.
The target loss and the distilled model train concurrently.

During testing/validation or when a post-sequential node queries us, we do not
recursively query our own child, instead, we use the distilled model as input to
our own network; this cuts the recursion and stops gradients from passing farther
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
        """Joins training thread and stops.
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
            self._run_graph(batch_x, batch_y, keep_prop=0.95, use_synthetic=False, do_train=True, do_metrics=False)
            self._run_graph(batch_x, batch_y, keep_prop=0.95, use_synthetic=True, do_train=True, do_metrics=False)

            # Run test step and print.
            if step % self._hparams.n_print == 0:

                # Train / Validation with and without synthetic models.
                tr_x, tr_y = self._mnist.train.next_batch(self._hparams.batch_size)
                val_x = self._mnist.test.images
                val_y = self._mnist.test.labels
                syn_tr_out = self._run_graph(tr_x, tr_y, keep_prop=1.0, use_synthetic=True, do_train=False, do_metrics=True)
                tr_out = self._run_graph(tr_x, tr_y, keep_prop=1.0, use_synthetic=False, do_train=False, do_metrics=True)
                syn_val_out = self._run_graph(val_x, val_y, keep_prop=1.0, use_synthetic=True, do_train=False, do_metrics=True)
                val_out = self._run_graph(val_x, val_y, keep_prop=1.0, use_synthetic=False, do_train=False, do_metrics=True)

                # Accuracy metrics.
                self._tblogger.log_scalar('validation accuracy using child', syn_val_out['accuracy'][0], step)
                self._tblogger.log_scalar('validation accuracy using synthetic', val_out['accuracy'][0], step)
                self._tblogger.log_scalar('training accuracy using synthetic', syn_tr_out['accuracy'][0], step)
                self._tblogger.log_scalar('training_accuracy using child', tr_out['accuracy'][0], step)

                # Target loss.
                self._tblogger.log_scalar('training target loss using synthetic', syn_tr_out['target_loss'][0], step)
                self._tblogger.log_scalar('training target loss using child', tr_out['target_loss'][0], step)

                # Synthetic loss.
                self._tblogger.log_scalar('training synthetic loss', tr_out['synthetic_loss'], step)
                self._tblogger.log_scalar('validation synthetic loss', val_out['synthetic_loss'], step)

                # Inputs score.
                self._tblogger.log_scalar('inputs pruning score', tr_out['inputs_pruning_score'][0], step)
                self._tblogger.log_scalar('inputs absolute sum of gradients', tr_out['inputs_absolute_gradient'][0], step)
                self._tblogger.log_scalar('inputs weight magnitude', tr_out['inputs_weight_magnitude'][0], step)

                # Downstream score.
                self._tblogger.log_scalar('downstream pruning score', tr_out['downstream_pruning_score'][0], step)
                self._tblogger.log_scalar('downstream absolute sum of gradients', tr_out['downstream_absolute_gradient'][0], step)
                self._tblogger.log_scalar('downstream weight magnitude', tr_out['downstream_weight_magnitude'][0], step)

                # Downstream Integrate gradients.
                self._tblogger.log_scalar('downstream integrated gradients score', syn_val_out['downstream_integrated_gradients_score'], step)
                self._tblogger.log_scalar('inputs integrated gradients score',  syn_val_out['input_integrated_gradients_score'], step)


                logger.info('{}: [val: {} - {}  tr: {} - {}]', self.name, val_out['accuracy'][0], syn_val_out['accuracy'][0], tr_out['accuracy'][0], syn_tr_out['accuracy'][0])
            if step > self._hparams.n_train_steps:
                self.running = False
            step+=1

    def _run_graph(self, spikes, targets, keep_prop, use_synthetic, do_train, do_metrics):
        """Runs the graph and returns fetch outputs.

        Args:
            spikes (numpy): mnist inputs [batch_size, 784]
            targets (numpy): mnist targets [batch_size, 10]
            keep_prop (float): dropout rate.
            use_synthetic (bool): do we use synthetic inputs or query child.
            do_train (bool): do we trigger training step.
        """

        # Query child if exists.
        cspikes =  np.zeros((np.shape(spikes)[0], self._hparams.n_embedding))
        if self._child and not use_synthetic:
            cspikes = self._child.spike(spikes)

        # Build feeds.
        feeds = {
            self._spikes: spikes, # Mnist 784 input.
            self._cspikes: cspikes, # Child inputs.
            self._targets: targets, # Mnist 1-hot Targets.
            self._use_synthetic: use_synthetic, # Do not use synthetic inputs.
            self._keep_rate: keep_prop, # Dropout.
        }

        # Build fetches.
        fetches = {}

        if do_train:
            fetches['target_step'] = self._tstep

        # We train the synthetic model when we query our child.
        if not use_synthetic:
            fetches['synthetic_loss'] = self._syn_loss # Distillation loss.

        if do_train:
            fetches['synthetic_step'] = self._syn_step # Synthetic step.
            fetches['child_gradients'] = self._tdgrads

        if do_metrics:
            fetches['accuracy'] = self._accuracy, # Classification accuracy.
            fetches['target_loss'] = self._target_loss, # Target accuracy.
            fetches['inputs_pruning_score'] = self._inputs_pruning_score, # Salience of inputs.
            fetches['inputs_weight_magnitude'] =  self._inputs_weight_magnitude,
            fetches['inputs_absolute_gradient'] = self._inputs_absolute_gradient,
            fetches['downstream_pruning_score'] = self._downstream_pruning_score, # Salience of downstream.
            fetches['downstream_weight_magnitude'] = self._downstream_weight_magnitude,
            fetches['downstream_absolute_gradient'] = self._downstream_absolute_gradient,
            fetches['input_integrated_gradients_score'] = self._input_ig
            fetches['downstream_integrated_gradients_score'] = self._downstream_ig

        # Run graph.
        run_output = self._session.run(fetches, feeds)

        # Pass the gradients to the child.
        if self._child and do_train and not use_synthetic:
            self._child.grade(spikes, run_output['child_gradients'])

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
        synthetic_network_variables = list(syn_weights.values()) + list(syn_biases.values())

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

        # Embedding: Apply the hidden layer to the spikes and embeddings from the previous component.
        # The embedding is the output for this component passed to its parents.
        self._input_layer = tf.concat([self._spikes, self._downstream], axis=1)
        hidden_layer1 = tf.nn.relu(tf.add(tf.matmul(self._input_layer, weights['w1']), biases['b1']))
        hidden_layer2 = tf.nn.relu(tf.add(tf.matmul(hidden_layer1, weights['w2']), biases['b2']))
        drop_hidden_layer2 = tf.nn.dropout(hidden_layer2, self._keep_rate)
        self._embedding = tf.nn.relu(tf.add(tf.matmul(drop_hidden_layer2, weights['w3']), biases['b3']))


        # Target: Apply a softmax over the embeddings. This is the loss from the local network.
        # The loss on the target and the loss from the parent averaged.
        self._logits = tf.add(tf.matmul(self._embedding, weights['w4']), biases['b4'])
        self._target_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._targets, logits=self._logits))

        # Metrics: We calcuate accuracy here because we are working with MNIST.
        correct = tf.equal(tf.argmax(self._logits, 1), tf.argmax(self._targets, 1))
        self._accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        # Optimizer: The optimizer for this component, could be different accross components.
        optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)

        # syn_grads: Gradient terms for the synthetic inputs.
        self._syn_grads = optimizer.compute_gradients(loss=self._syn_loss + self._target_loss, var_list=synthetic_network_variables)

        # Embedding grads: Here, we compute the gradient terms for the embedding with respect
        # to the gradients passed from the parent (a.k.a egrads). Dgrads is the gradient for
        # the downstream component (child) and elgrads are the gradient terms for the the local
        # FFNN.
        self._cgrads = optimizer.compute_gradients(loss=self._embedding, var_list=[self._cspikes], grad_loss=self._egrads)[0][0]
        self._elgrads = optimizer.compute_gradients(loss=self._embedding, var_list=local_network_variables, grad_loss=self._egrads)

        # Gradients from target: Here, we compute the gradient terms for the downstream child and
        # the local variables but with respect to the target loss. These get sent downstream and used to
        # optimize the local variables.
        self._tdgrads = optimizer.compute_gradients(loss=self._target_loss, var_list=[self._cspikes])[0][0]
        self._tlgrads = optimizer.compute_gradients(loss=self._target_loss, var_list=local_network_variables)

        # Syn step: Train step which applies the synthetic input grads to the synthetic input model.
        self._syn_step = optimizer.apply_gradients(self._syn_grads)

        # Embedding trainstep: Train step which applies the gradients calculated w.r.t the gradients
        # from a parent.
        self._estep = optimizer.apply_gradients(self._elgrads)

        # Target trainstep: Train step which applies the gradients calculated w.r.t the target loss.
        self._tstep = optimizer.apply_gradients(self._tlgrads)

        # Compute information scores for raw input.
        self._inputs_grads = optimizer.compute_gradients(loss=self._target_loss, var_list=self._spikes)
        self._inputs_absolute_gradient = tf.reduce_sum(tf.reduce_sum(tf.abs(self._inputs_grads[0][0])))
        self._inputs_pruning_score = self._pruning_score(self._spikes, self._inputs_grads[0][0])
        self._inputs_weight_magnitude = tf.reduce_sum(tf.reduce_sum(tf.slice(weights['w1'], [0, 0], [784, -1])))

        # Compute information scores for downstream component.
        self._downstream_grads = optimizer.compute_gradients(loss=self._target_loss, var_list=self._downstream)
        self._downstream_absolute_gradient = tf.reduce_sum(tf.reduce_sum(tf.abs(self._downstream_grads[0][0])))
        self._downstream_pruning_score = self._pruning_score(self._downstream, self._downstream_grads[0][0])
        self._downstream_weight_magnitude = tf.reduce_sum(tf.reduce_sum(tf.slice(weights['w1'], [784, 0], [-1, -1])))

        # integrate gradient score
        integrated_gradients = tf.abs(self._integrated_gradients(self._logits, self._input_layer, 10))
        total_ig = tf.reduce_sum(integrated_gradients)
        downstream_ig = tf.reduce_sum(tf.slice(integrated_gradients, [0, 784], [-1, -1]))
        input_ig = tf.reduce_sum(tf.slice(integrated_gradients, [0, 0], [-1, 784]))
        self._downstream_ig = downstream_ig/total_ig
        self._input_ig = input_ig/total_ig

    def _integrated_gradients(self, target, input_tensor, steps):
        baseline = tf.zeros_like(input_tensor)
        scaled_inputs = [baseline + (float(i)/steps)*(input_tensor - baseline) for i in range(0, steps+1)]

        grads = []
        for next_inp in scaled_inputs:
            grads.append(tf.gradients(ys=target, xs=input_tensor)[0])

        trapazoidal_grads = []
        for i in range(len(grads)-1):
            trapazoidal_grads.append((grads[i] + grads[i+1])/2)

        avg_trapazoidal = tf.add_n(trapazoidal_grads)/len(grads)
        avg_grads = tf.reduce_mean(avg_trapazoidal, axis=0)
        integrated_gradients = (input_tensor - baseline) * avg_grads  # shape: <inp.shape>

        return integrated_gradients

    def _pruning_score(self, value, gradient):
        g = tf.tensordot(-value, gradient, axes=2)
        gxgx = tf.multiply(gradient, gradient)
        H = tf.tensordot(-value, gxgx, axes=2)
        score = tf.reduce_sum(g + H)
        return score