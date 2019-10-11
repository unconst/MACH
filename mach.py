class Config:
    def __init__(self):
        self.n_input = 784  # input layer (28x28 pixels)
        self.n_hidden1 = 512  # 1st hidden layer
        self.n_hidden2 = 256  # 2nd hidden layer
        self.n_embedding = 128  # 3rd embedding layer
        self.n_output = 10  # output layer (0-9 digits)
        self.n_model_inputs = self.n_input + self.n_embedding  # input layer (28x28 pixels)

        self.learning_rate = 1e-4
        self.learn_iterations = 1000
        self.batch_size = 128
        self.mach_batch_size = 512

class Mach:

    def __init__(self, config, mnist):
        self.c = config
        self.mnist = mnist
        self.n_train = self.mnist.train.num_examples  # 55,000
        self.n_validation = self.mnist.validation.num_examples  # 5000
        self.n_test = self.mnist.test.num_examples  # 10,000

        self.gradient_queue = queue.LifoQueue(maxsize=-1)

        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.input_fn()
            self.model_fn()
            self.grad_fn()
            init = tf.global_variables_initializer()
        self.session.run(init)

        self.child = None

    def input_fn(self):
        # Placeholders.
        self.X = tf.placeholder(tf.float32, [None, self.c.n_input], 'X')
        self.Y = tf.placeholder(tf.float32, [None, self.c.n_output], 'Y')
        self.C = tf.placeholder(tf.float32, [None, self.c.n_embedding], 'C')

    def grad_fn(self):
        # Optimizer.
        self.optimizer = tf.train.AdamOptimizer(self.c.learning_rate)

        # Gradients.
        self.E_grad = tf.placeholder(tf.float32, [None, self.c.n_embedding], 'E')
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=self.logits))
        self.upstream_gradients = self.optimizer.compute_gradients(loss=self.cross_entropy)
        self.local_gradients = self.optimizer.compute_gradients(loss=self.cross_entropy)

        # Secondary metrics.
        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Upstream Gradient placeholders.
        self.upstream_gradient_values = []
        self.upstream_placeholder_gradients = []
        for gradient_variable in self.upstream_gradients:
            grad_placeholder = tf.placeholder(tf.float32, shape=gradient_variable[1].get_shape())
            self.upstream_gradient_values.append(gradient_variable[1])
            self.upstream_placeholder_gradients.append((grad_placeholder, gradient_variable[1]))

        # Local Gradient placeholders.
        self.local_gradient_values = []
        self.local_placeholder_gradients = []
        for gradient_variable in self.local_gradients:
            grad_placeholder = tf.placeholder(tf.float32, shape=gradient_variable[1].get_shape())
            self.local_gradient_values.append(gradient_variable[1])
            self.local_placeholder_gradients.append((grad_placeholder, gradient_variable[1]))

        # Train steps.
        self.upstream_train_step = self.optimizer.apply_gradients(self.upstream_placeholder_gradients)
        self.local_train_step = self.optimizer.apply_gradients(self.local_placeholder_gradients)


    def model_fn(self):
        self.model_inputs = tf.concat([self.X, self.C], axis=1)

        weights = {
            'w1': tf.Variable(tf.truncated_normal([self.c.n_model_inputs, self.c.n_hidden1], stddev=0.1)),
            'w2': tf.Variable(tf.truncated_normal([self.c.n_hidden1, self.c.n_hidden2], stddev=0.1)),
            'w3': tf.Variable(tf.truncated_normal([self.c.n_hidden2, self.c.n_embedding], stddev=0.1)),
            'out': tf.Variable(tf.truncated_normal([self.c.n_embedding, self.c.n_output], stddev=0.1)),
        }

        biases = {
            'b1': tf.Variable(tf.constant(0.1, shape=[self.c.n_hidden1])),
            'b2': tf.Variable(tf.constant(0.1, shape=[self.c.n_hidden2])),
            'b3': tf.Variable(tf.constant(0.1, shape=[self.c.n_embedding])),
            'out': tf.Variable(tf.constant(0.1, shape=[self.c.n_output]))
        }

        # Model feature extraction.
        self.layer_1 = tf.add(tf.matmul(self.model_inputs, weights['w1']), biases['b1'])
        self.layer_2 = tf.add(tf.matmul(self.layer_1, weights['w2']), biases['b2'])
        self.E = tf.add(tf.matmul(self.layer_2, weights['w3']), biases['b3'])

        # Local logits.
        self.logits = tf.matmul(self.E, weights['out']) + biases['out']

    def Child(self, batch):
        if self.child == None:
            return numpy.zeros((numpy.shape(batch)[0], self.c.n_embedding))
        else:
            return self.child.Spike(batch)

    def Spike(self, spikes):
        feeds={self.X: spikes, self.C: self.Child(spikes), self.keep_prob: 1.0}
        return self.session.run([self.E], feed_dict=feeds)[0]

    def Grade(self, grads, spikes):
        cspikes = self.Child(spikes)
        feeds={
                self.X: batch,
                self.C: self.Child(batch),
                self.E_grad: grads,
                self.keep_prob: 1.0
        }
        fetches = [self.upstream_gradient_values]
        upstream_gradients = self.session.run(fetches, feeds)[0]
        self.grad_queue.put(upstream_gradients)


    def Train(self):
        # train on mini batches
        for i in range(self.c.learn_iterations):
            batch_step()

    def batch_step(self):
        batch_x, batch_y = self.mnist.train.next_batch(self.c.batch_size)
        feeds={self.X: batch_x, self.Y: batch_y, self.C: self.Child(batch_x)}
        fetches=self.local_gradients
        gradients = self.session.run(self.local_gradients, feed_dict=feeds)
        self.gradient_queue.put(gradients)

    def learn_step(self):
        gradients = self.gradient_queue.get()
        feeds = {}
        for j, grad_var in enumerate(gradients):
            feeds[self.local_placeholder_gradients[j][0]] = gradients[j][0]
        self.session.run(self.local_train_step, feeds)

    # # print loss and accuracy (per minibatch)
    # if i % 100 == 0:
    #     fetches = [self.cross_entropy, self.accuracy]
    #     feeds = {self.X: batch_x, self.Y: batch_y, self.C: self.Child(batch_x)}
    #     minibatch_loss, minibatch_accuracy = self.session.run(fetches,feeds)
    #     print(
    #         "Iteration",
    #         str(i),
    #         "\t| Loss =",
    #         str(minibatch_loss),
    #         "\t| Accuracy =",
    #         str(minibatch_accuracy)
    #         )

    def Test(self):
        feed_dict = {
                self.X: self.mnist.test.images,
                self.Y: self.mnist.test.labels,
                self.C: self.Child(self.mnist.test.images),
                self.keep_prob: 1.0
        }
        test_accuracy = self.session.run(self.accuracy, feed_dict=feed_dict)
        print("\nAccuracy on test set:", test_accuracy)
