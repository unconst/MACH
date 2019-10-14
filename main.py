import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def load_data_and_constants(hparams):

    # Load mnist data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    hparams.n_inputs = 784  # input layer (28x28 pixels)
    hparams.n_embedding = 128  # 3rd embedding layer
    hparams.n_targets = 10  # output layer (0-9 digits)

    hparams.t_hidden1 = 512  # 1st hidden layer
    hparams.t_hidden2 = 256  # 2nd hidden layer

    hparams.s_hidden1 = 512  # 1st hidden layer
    hparams.s_hidden2 = 256  # 2nd hidden layer

    return mnist, hparams

def inputs(hparams):
    x_inputs = tf.placeholder("float", [None, hparams.n_inputs])
    y_targets = tf.placeholder("float", [None, hparams.n_targets])
    return x_inputs, y_targets

def teacher(x_inputs, hparams):

    weights = {
        'w1': tf.Variable(tf.truncated_normal([hparams.n_inputs, hparams.t_hidden1], stddev=0.1)),
        'w2': tf.Variable(tf.truncated_normal([hparams.t_hidden1, hparams.t_hidden2], stddev=0.1)),
        'w3': tf.Variable(tf.truncated_normal([hparams.t_hidden2, hparams.n_embedding], stddev=0.1)),
    }

    biases = {
        'b1': tf.Variable(tf.constant(0.1, shape=[hparams.t_hidden1])),
        'b2': tf.Variable(tf.constant(0.1, shape=[hparams.t_hidden2])),
        'b3': tf.Variable(tf.constant(0.1, shape=[hparams.n_embedding])),
    }

    layer_1 = tf.nn.relu(tf.add(tf.matmul(x_inputs, weights['w1']), biases['b1']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['w2']), biases['b2']))
    teacher_embedding = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['w3']), biases['b3']))

    return teacher_embedding

def student(x_inputs, hparams):

    weights = {
        'w1': tf.Variable(tf.truncated_normal([hparams.n_inputs, hparams.s_hidden1], stddev=0.1)),
        'w2': tf.Variable(tf.truncated_normal([hparams.s_hidden1, hparams.s_hidden2], stddev=0.1)),
        'w3': tf.Variable(tf.truncated_normal([hparams.s_hidden2, hparams.n_embedding], stddev=0.1)),
    }

    biases = {
        'b1': tf.Variable(tf.constant(0.1, shape=[hparams.s_hidden1])),
        'b2': tf.Variable(tf.constant(0.1, shape=[hparams.s_hidden2])),
        'b3': tf.Variable(tf.constant(0.1, shape=[hparams.n_embedding])),
    }

    layer_1 = tf.nn.relu(tf.add(tf.matmul(x_inputs, weights['w1']), biases['b1']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['w2']), biases['b2']))
    student_embedding = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])

    return student_embedding

def logits(t_embedding, s_embedding, hparams):
    w = tf.Variable(tf.truncated_normal([hparams.n_embedding, hparams.n_targets], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[hparams.n_targets])),
    t_logits = tf.add(tf.matmul(t_embedding, w), b)
    s_logits = tf.add(tf.matmul(s_embedding, w), b)
    return t_logits, s_logits

def target_loss(logits, targets, hparams):
    target_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2( labels=targets, logits=logits))
    return target_loss

def distillation_loss(student_embedding, teacher_embedding, hparams):
    distillation_loss = tf.reduce_mean(tf.nn.l2_loss(tf.stop_gradient(teacher_embedding) - student_embedding))
    return distillation_loss

def step_and_metrics(teacher_embedding,
                    student_embedding,
                    teacher_logits,
                    student_logits,
                    teacher_loss,
                    student_loss,
                    dist_loss,
                    y_targets,
                    x_inputs,
                    hparams):

    full_loss = teacher_loss + dist_loss

    step = tf.train.AdamOptimizer(hparams.learning_rate).minimize(full_loss)

    t_correct = tf.equal(tf.argmax(teacher_logits, 1), tf.argmax(y_targets, 1))
    t_accuracy = tf.reduce_mean(tf.cast(t_correct, tf.float32))

    s_correct = tf.equal(tf.argmax(student_logits, 1), tf.argmax(y_targets, 1))
    s_accuracy = tf.reduce_mean(tf.cast(s_correct, tf.float32))

    metrics = {
            'tloss' : teacher_loss,
            'sloss' : student_loss,
            'tacc'  : t_accuracy,
            'sacc'  : s_accuracy,
    }

    return step, metrics


def main(hparams):

    # Best:

    mnist, hparams = load_data_and_constants(hparams)

    # Build graph and session.
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default():

        x_inputs, y_targets = inputs(hparams)

        teacher_embedding = teacher(x_inputs, hparams)

        student_embedding = student(x_inputs, hparams)

        teacher_logits, student_logits = logits(teacher_embedding, student_embedding, hparams)

        teacher_loss = target_loss(teacher_logits, y_targets, hparams)

        student_loss = target_loss(tf.stop_gradient(student_logits), y_targets, hparams)

        dist_loss = distillation_loss(student_embedding, teacher_embedding, hparams)

        step, metrics = step_and_metrics(   teacher_embedding,
                                            student_embedding,
                                            teacher_logits,
                                            student_logits,
                                            teacher_loss,
                                            student_loss,
                                            dist_loss,
                                            y_targets,
                                            x_inputs,
                                            hparams)

        session.run(tf.global_variables_initializer())


    # Training loop.
    for i in range(hparams.n_iterations):
        batch_x, batch_y = mnist.train.next_batch(hparams.batch_size)
        feeds = {
            x_inputs: batch_x,
            y_targets: batch_y
        }
        session.run(step, feeds)

        # print loss and accuracy (per minibatch)
        if i % 100 == 0:
            feeds = {
                x_inputs: batch_x,
                y_targets: batch_y
            }
            train_metrics = session.run(metrics, feeds)
            print(train_metrics)

    # Test.
    feeds = {
        x_inputs: mnist.test.images,
        y_targets: mnist.test.labels
    }
    test_metrics = session.run(metrics, feeds)
    print("\nTeacher accuracy on test set:", test_metrics['tacc'])
    print("\nStudent accuracy on test set:", test_metrics['sacc'])


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--batch_size',
                      default=128,
                      type=int,
                      help='The number of examples per batch. Default batch_size=128')
  parser.add_argument('--n_iterations',
                      default=10000,
                      type=int,
                      help='The number of training iterations. Default iterations=10000')
  parser.add_argument('--learning_rate',
                    default=1e-4,
                    type=float,
                    help='Network learning rate. Default learning_rate=1e-4')

  hparams = parser.parse_args()

  main(hparams)
