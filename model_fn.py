
class Modelfn():

    def __init__(self, hparams):
        self._hparams = hparams

    def _gate_dispatch(self, spikes):
        raise NotImplementedError

    def _gate_combine(self, spikes):
        raise NotImplementedError

    def _tokenizer_network(self, x_batch):

        # Tokenization with lookup table. Retrieves a 1 x vocabulary sized
        # vector.
        vocabulary_table = tf.contrib.lookup.index_table_from_tensor(
            mapping=tf.constant(self._string_map),
            num_oov_buckets=1,
            default_value=0)

        # Token embedding matrix is a matrix of vectors. During lookup we pull
        # the vector corresponding to the 1-hot encoded vector from the
        # vocabulary table.
        embedding_matrix = tf.Variable(
            tf.random.uniform([self._hparams.n_vocabulary, self._hparams.n_embedding], -1.0,
                              1.0))

        # Tokenizer network.
        x_batch = tf.reshape(x_batch, [-1])

        # Apply tokenizer lookup.
        x_batch = vocabulary_table.lookup(x_batch)

        # Apply table lookup to retrieve the embedding.
        x_batch = tf.nn.embedding_lookup(embedding_matrix, x_batch)
        x_batch = tf.reshape(x_batch, [-1, self._hparams.n_embedding])

        raise x_batch

    def _synthetic_network(self, tokenized_spikes):
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

        syn_hidden1 = tf.nn.relu(tf.add(tf.matmul(tokenized_spikes, syn_weights['syn_w1']), syn_biases['syn_b1']))
        syn_hidden2 = tf.nn.relu(tf.add(tf.matmul(syn_hidden1, syn_weights['syn_w2']), syn_biases['syn_b2']))
        syn_embedding = tf.add(tf.matmul(syn_hidden2, syn_weights['syn_w3']), syn_biases['syn_b3'])
        return syn_embedding

    def _embedding_network(self, token_embedding, downstream_embedding):
        # Weights and biases
        embd_weights = {
            'embedding_w1': tf.Variable(tf.random.truncated_normal([self._hparams.n_inputs + self._hparams.n_embedding, self._hparams.n_hidden1], stddev=0.1)),
            'embedding_w2': tf.Variable(tf.random.truncated_normal([self._hparams.n_hidden1, self._hparams.n_hidden2], stddev=0.1)),
            'embedding_w3': tf.Variable(tf.random.truncated_normal([self._hparams.n_hidden2, self._hparams.n_embedding], stddev=0.1)),
            'embedding_w4': tf.Variable(tf.random.truncated_normal([self._hparams.n_embedding, self._hparams.n_targets], stddev=0.1)),
            }
        embd_biases = {
            'embedding_b1': tf.Variable(tf.constant(0.1, shape=[self._hparams.n_hidden1])),
            'embedding_b2': tf.Variable(tf.constant(0.1, shape=[self._hparams.n_hidden2])),
            'embedding_b3': tf.Variable(tf.constant(0.1, shape=[self._hparams.n_embedding])),
            'embedding_b4': tf.Variable(tf.constant(0.1, shape=[self._hparams.n_targets])),
        }
        input_layer = tf.concat([token_embedding, downstream_embedding], axis=1)
        embed_layer1 = tf.nn.relu(tf.add(tf.matmul(input_layer, weights['w1']), biases['b1']))
        embed_layer2 = tf.nn.relu(tf.add(tf.matmul(hidden_layer1, weights['w2']), biases['b2']))
        embed_output = tf.nn.relu(tf.add(tf.matmul(drop_hidden_layer2, weights['w3']), biases['b3']))
        return embed_output

    def _target_network(self, embedding_spikes):
        weights = {
            'w1': tf.Variable(tf.random.truncated_normal([self._hparams.n_embedding, self._hparams.n_targets], stddev=0.1)),
            }
        biases = {
            'b1': tf.Variable(tf.constant(0.1, shape=[self._hparams.n_targets])),
        }
        logits = tf.add(tf.matmul(embedding_spikes, weights['w1']), biases['b1'])
        raise logits

    def _target_loss(self, targets, logits):
        target_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets, logits=logits))
        raise target_loss

    def _synthetic_loss(self, embedding_spikes):
        raise NotImplementedError

    def _model_fn(self):

        # Spikes: inputs from the dataset of arbitrary batch_size.
        self.spikes = tf.compat.v1.placeholder(tf.string, [None, 1], name='spikes')

        # Parent gradients: Gradients passed by this components parent.
        self.parent_error = tf.compat.v1.placeholder(tf.float32, [None, self._hparams.n_embedding], name='parent_grads')

        # Targets: Supervised signals used during training and testing.
        self.targets = tf.compat.v1.placeholder(tf.float32, [None, self._hparams.n_targets], name='targets')

        # Use Synthetic: Flag, use synthetic inputs when running graph.
        self.use_synthetic = tf.compat.v1.placeholder(tf.bool, shape=[], name='use_synthetic')

        # Gating network.
        with tf.compat.v1.variable_scope("gating_network"):
            gated_spikes = self._gate_dispatch(self.spikes)
            child_inputs = []
            for i, gated_spikes in enumerate(gated_spikes):
                child_inputs.append(_input_from_gate(gated_spikes))
            child_spikes = self._gate_combine(child_inputs)
            gating_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="gating_network")

        # Tokenizer network.
        with tf.compat.v1.variable_scope("tokenizer_network"):
            tokenized_spikes = self._tokenizer(self.spikes)
            tokenizer_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="tokenizer_network")

        # Synthetic network.
        with tf.compat.v1.variable_scope("synthetic_network"):
            synthetic_spikes = self._synthetic_network(tokenized_spikes)
            synthetic_loss = self._synthetic_loss(synthetic_spikes, self.child_spikes)
            synthetic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="synthetic_network")

        # Downstream switch
        downstream_spikes = tf.cond(
            tf.equal(self.use_synthetic, tf.constant(True)),
            true_fn=lambda: synthetic_spikes,
            false_fn=lambda: child_spikes)

        # Embedding network.
        with tf.compat.v1.variable_scope("embedding_network"):
            self.embedding = self._embedding_network(tokenized_spikes, downstream_spikes)
            embedding_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="embedding_network")

        # Target network
        with tf.compat.v1.variable_scope("target_network"):
            logits = self._target_network(self.embedding)
            target_loss = self._target_loss(logits, self.targets)
            target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_network")


        # Optimizer
        optimizer = self._optimizer()

        # Synthetic grads.
        synthetic_grads = optimizer.compute_gradients(  loss = synthetic_loss,
                                                        var_list = synthetic_vars)

        # Parent grads
        parent_grads = optimizer.compute_gradients(    loss = self.embedding,
                                                       var_list = embedding_vars,
                                                       grad_loss = self.parent_error)

        # Target grads
        target_grads = optimizer.compute_gradients(    loss = target_loss,
                                                       var_list = target_vars + embedding_vars + gate_vars)

        # Child grads
        child_grads = optimizer.compute_gradients(  loss = target_loss,
                                                    var_list = child_inputs)

        # Synthetic step.
        synthetic_step = optimizer.apply_gradients(synthetic_grads)

        # Parent step.
        parent_step = optimizer.apply_gradients(parent_grads)

        # Target step.
        target_step = optimizer.apply_gradients(target_grads)


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
        self._expert_inputs = dispatcher.dispatch(self._spikes)

        # Join expert inputs.
        self._expert_outputs = []
        for i in range(self._hparams.n_components - 1):
            self._expert_outputs.append(tf.compat.v1.placeholder_with_default(tf.zeros([tf.shape(self._expert_inputs[i])[0], self._hparams.n_embedding]), [None, self._hparams.n_embedding], name='einput' + str(i)))

        # Child spikes if needed.
        self._cspikes = dispatcher.combine(self._expert_outputs)


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

    # Optimizer: The optimizer for this component, could be different accross components.
    optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)

    # syn_grads: Gradient terms for the synthetic inputs.
    self._syn_grads = optimizer.compute_gradients(loss=self._syn_loss + self._target_loss, var_list=synthetic_network_variables)

    # Embedding grads: Here, we compute the gradient terms for the embedding with respect
    # to the gradients passed from the parent (a.k.a egrads). Dgrads is the gradient for
    # the downstream component (child) and elgrads are the gradient terms for the the local
    # FFNN.
    self._cgrads = optimizer.compute_gradients(loss=self._embedding, var_list=self._expert_outputs, grad_loss=self._egrads)[0][0]
    self._elgrads = optimizer.compute_gradients(loss=self._embedding, var_list=local_network_variables, grad_loss=self._egrads)

    # Gradients from target: Here, we compute the gradient terms for the downstream child and
    # the local variables but with respect to the target loss. These get sent downstream and used to
    # optimize the local variables.
    self._tdgrads = optimizer.compute_gradients(loss=self._target_loss, var_list=self._expert_outputs)
    self._tlgrads = optimizer.compute_gradients(loss=self._target_loss, var_list=local_network_variables)
    self._tggrads = optimizer.compute_gradients(loss=self._target_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="gate"))

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
