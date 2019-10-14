from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy
import queue

import tensorflow as tf
import numpy
import queue


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
        self.session = tf.compat.v1.Session(graph=self.graph)
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
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.c.learning_rate)

        # Loss.
        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y,
                                                       logits=self.logits))

        # Secondary loss metrics.
        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Gradients.
        self.E_grad = tf.placeholder(tf.float32, [None, self.c.n_embedding],
                                     'E')
        self.upstream_gradients = self.optimizer.compute_gradients(
            loss=self.logits, grad_loss=self.E_grad)
        self.local_gradients = self.optimizer.compute_gradients(
            loss=self.cross_entropy)
        self.C_grad = tf.gradients(xs=self.C, ys=self.cross_entropy)

        # 1. Upstream Gradient placeholders.
        self.upstream_placeholder_gradients = []
        for gradient_variable in self.upstream_gradients:
            grad_placeholder = tf.placeholder(
                tf.float32, shape=gradient_variable[1].get_shape())
            self.upstream_placeholder_gradients.append(
                (grad_placeholder, gradient_variable[1]))

        # 2. Local Gradient placeholders.
        self.local_placeholder_gradients = []
        for gradient_variable in self.local_gradients:
            grad_placeholder = tf.placeholder(
                tf.float32, shape=gradient_variable[1].get_shape())
            self.local_placeholder_gradients.append(
                (grad_placeholder, gradient_variable[1]))

        # Train steps.
        self.upstream_train_step = self.optimizer.apply_gradients(
            self.upstream_placeholder_gradients)
        self.local_train_step = self.optimizer.apply_gradients(
            self.local_placeholder_gradients)

    def model_fn(self):
        self.model_inputs = tf.concat([self.X, self.C], axis=1)

        weights = {
            'w1':
                tf.Variable(
                    tf.truncated_normal(
                        [self.c.n_model_inputs, self.c.n_hidden1], stddev=0.1)),
            'w2':
                tf.Variable(
                    tf.truncated_normal([self.c.n_hidden1, self.c.n_hidden2],
                                        stddev=0.1)),
            'w3':
                tf.Variable(
                    tf.truncated_normal([self.c.n_hidden2, self.c.n_embedding],
                                        stddev=0.1)),
            'out':
                tf.Variable(
                    tf.truncated_normal([self.c.n_embedding, self.c.n_output],
                                        stddev=0.1)),
        }

        biases = {
            'b1': tf.Variable(tf.constant(0.1, shape=[self.c.n_hidden1])),
            'b2': tf.Variable(tf.constant(0.1, shape=[self.c.n_hidden2])),
            'b3': tf.Variable(tf.constant(0.1, shape=[self.c.n_embedding])),
            'out': tf.Variable(tf.constant(0.1, shape=[self.c.n_output]))
        }

        # Model feature extraction.
        self.layer_1 = tf.add(tf.matmul(self.model_inputs, weights['w1']),
                              biases['b1'])
        self.layer_2 = tf.add(tf.matmul(self.layer_1, weights['w2']),
                              biases['b2'])
        self.E = tf.add(tf.matmul(self.layer_2, weights['w3']), biases['b3'])

        # Local logits.
        self.logits = tf.matmul(self.E, weights['out']) + biases['out']

    def Child(self, batch):
        if self.child == None:
            return numpy.zeros((numpy.shape(batch)[0], self.c.n_embedding))
        else:
            return self.child.Spike(batch)

    def Spike(self, spikes):
        feeds = {
            self.X: spikes,
            self.C: self.Child(spikes),
            self.keep_prob: 1.0
        }
        return self.session.run([self.E], feed_dict=feeds)[0]

    def Grade(self, grads, spikes):
        cspikes = self.Child(spikes)
        feeds = {self.X: batch, self.C: cspikes, self.E_grad: grads}
        fetches = [self.upstream_gradients]
        upstream_gradients = self.session.run(fetches, feeds)[0]
        self.grad_queue.put(upstream_gradients)

    def Perf(self):
        batch_x, batch_y = self.mnist.train.next_batch(self.c.batch_size)
        fetches = [self.cross_entropy, self.accuracy]
        feeds = {self.X: batch_x, self.Y: batch_y, self.C: self.Child(batch_x)}
        loss, accuracy = self.session.run(fetches, feeds)
        print("Loss =", str(loss), "\t| Accuracy =", str(accuracy))

    def Train(self, n):
        cgrads_0 = self.batch_step()
        for i in range(n):
            cgrads_i = self.batch_step()
            for j, grad in enumerate(cgrads_i):
                cgrads_0[j] += grad
        return cgrads_0

    def batch_step(self):
        batch_x, batch_y = self.mnist.train.next_batch(self.c.batch_size)
        feeds = {self.X: batch_x, self.Y: batch_y, self.C: self.Child(batch_x)}
        gradients = self.session.run([self.local_gradients, self.C_grad],
                                     feed_dict=feeds)
        self.gradient_queue.put(gradients[0])
        return gradients[1][0]

    def Learn(self, n):
        grad_avg = self.grad_avg(n)
        self.learn_step(grad_avg)

    def grad_avg(self, n):
        grads = self.gradient_queue.get()
        gradients_0 = [grad_var[0] for grad_var in grads]
        for i in range(n - 1):
            gradients_i = [
                grad_var[0] for grad_var in self.gradient_queue.get()
            ]
            for j, grad in enumerate(gradients_i):
                gradients_0[j] += grad
        for i in range(len(gradients_0)):
            gradients_0[0] /= n
        return gradients_0

    def learn_step(self, grads):
        feeds = {}
        for j, grad_var in enumerate(grads):
            feeds[self.local_placeholder_gradients[j][0]] = grad_var

        self.session.run(self.local_train_step, feeds)

    def Test(self):
        feed_dict = {
            self.X: self.mnist.test.images,
            self.Y: self.mnist.test.labels,
            self.C: self.Child(self.mnist.test.images)
        }
        test_accuracy = self.session.run(self.accuracy, feed_dict=feed_dict)
        print("\nAccuracy on test set:", test_accuracy)



mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

if __name__ == "__main__":
    c = mach.Config()
    m = mach.Mach(c, mnist)

    for i in range(1000):
        m.Train(1)
        m.Learn(1)
        if i % 100 == 0:
            m.Test()
