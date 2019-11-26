from loguru import logger
import numpy as np
import tensorflow as tf
import threading

from utils import next_nounce

class Mach:
    def __init__(self, name, mnist, hparams, tblogger):
        self.name = name
        self._mnist = mnist
        self._hparams = hparams
        self._tblogger = tblogger
        self._child = None
        self._graph = tf.Graph()
        self._session = tf.compat.v1.Session(graph=self._graph)
        with self._graph.as_default():
            self._model_fn()
            self._session.run(tf.compat.v1.global_variables_initializer())

    def set_child(self, child):
        self._child = child

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        logger.info('starting thread {}', self.name)
        self._thread.start()

    def stop(self):
        self._running = False
        logger.info('joining thread {}', self.name)
        self._thread.join()

    def _run(self):
        step = 0
        while self._running:
            batch_x, batch_y = self._mnist.train.next_batch(self._hparams.batch_size)
            self._train(batch_x, batch_y)
            if step % self._hparams.n_print == 0:
                train_acc = self._train(batch_x, batch_y)
                val_acc = self._test(self._mnist.test.images, self._mnist.test.labels)
                self._tblogger.log_scalar('val_accuracy', val_acc, step)
                self._tblogger.log_scalar('tr_accuracy', train_acc, step)
                logger.info('{}: train {}, validation {}', self.name, train_acc, val_acc)
            step+=1

    def _test(self, spikes, targets):
        dspikes = np.zeros((np.shape(spikes)[0], self._hparams.n_embedding))
        feeds = {
            self._spikes: spikes,
            self._dspikes: dspikes,
            self._targets: targets,
            self._use_synthetic: True,
            self._keep_rate: 1.0,
        }
        return self._session.run(self._accuracy, feeds)

    def _train(self, spikes, targets):
        # Computes the output for this node with respect to input spikes.
        # Calculates a target loss. Computes local and downstream gradients from
        # this loss. Send the downstream gradients to children and uses the local
        # gradients to apply a step.
        nounce = next_nounce()

        # Query child with increased depth.
        if self._child:
            dspikes = self._child.spike(nounce, spikes, 0)
        else:
            dspikes = np.zeros((np.shape(spikes)[0], self._hparams.n_embedding))

        feeds = {
            self._spikes: spikes,
            self._dspikes: dspikes,
            self._targets: targets,
            self._use_synthetic: False,
            self._keep_rate: 0.96,
        }
        # Compute the target loss from the input and make a training step.
        fetches = [self._tdgrads, self._accuracy, self._tstep, self._syn_loss, self._syn_step]
        run_output = self._session.run(fetches, feeds)

        # Recursively pass the gradients through the graph.
        if self._child:
            self._child.grade(nounce, spikes, run_output[0], 0)

        # Return the batch accuracy.
        return run_output[1]

    def spike(self, nounce, spikes, depth):
        # Return using synthetic inputs as input to this component.
        zeros = np.zeros((np.shape(spikes)[0], self._hparams.n_embedding))
        feeds = {
            self._spikes: spikes,
            self._dspikes: zeros,
            self._use_synthetic: True,
            self._keep_rate: 1.0,
        }
        return self._session.run(self._embedding, feeds)

    def grade(self, nounce, spikes, grads, depth):
        # Computes the gradients for the local node as well as the gradients
        # for its children. Applies the gradients to the local node and sends the
        # children gradients downstream.
        zeros = np.zeros((np.shape(spikes)[0], self._hparams.n_embedding))
        feeds = {
            self._spikes: spikes,
            self._egrads: grads,
            self._dspikes: zeros,
            self._use_synthetic: False,
            self._keep_rate: 1.0
        }
        # Compute gradients for the children and apply the local step.
        self._session.run([self._estep], feeds)

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
        syn_dspikes = tf.add(tf.matmul(syn_hidden2, syn_weights['syn_w3']), syn_biases['syn_b3'])
        #syn_dspikes = tf.Print(syn_dspikes, [self._dspikes, syn_dspikes], summarize=100000)
        self._syn_loss = tf.reduce_mean(tf.nn.l2_loss(tf.stop_gradient(self._dspikes) - syn_dspikes))
        tf.compat.v1.summary.scalar("syn_loss", self._syn_loss)

        # Switch between synthetic embedding or true_embedding
        dspikes = tf.cond(tf.equal(self._use_synthetic, tf.constant(True)),
                              true_fn=lambda: tf.stop_gradient(syn_dspikes),
                              false_fn=lambda: self._dspikes)

        # Embedding: Apply the hidden layer to the spikes and embeddings from the previous component.
        # The embedding is the output for this component passed to its parents.
        input_layer = tf.concat([self._spikes, dspikes], axis=1)
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
        self._dgrads = optimizer.compute_gradients(loss=self._embedding, var_list=[self._dspikes], grad_loss=self._egrads)[0][0]
        self._elgrads = optimizer.compute_gradients(loss=self._embedding, var_list=tf.compat.v1.trainable_variables(), grad_loss=self._egrads)

        # Gradients from target: Here, we compute the gradient terms for the downstream child and
        # the local variables but with respect to the target loss. These get sent downstream and used to
        # optimize the local variables.
        self._tdgrads = optimizer.compute_gradients(loss=target_loss, var_list=[self._dspikes])[0][0]
        self._tlgrads = optimizer.compute_gradients(loss=target_loss, var_list=tf.compat.v1.trainable_variables())

        # Syn step: Train step which applies the synthetic input grads to the synthetic input model.
        self._syn_step = optimizer.apply_gradients(self._syn_grads)

        # Embedding trainstep: Train step which applies the gradients calculated w.r.t the gradients
        # from a parent.
        self._estep = optimizer.apply_gradients(self._elgrads)

        # Target trainstep: Train step which applies the gradients calculated w.r.t the target loss.
        self._tstep = optimizer.apply_gradients(self._tlgrads)
