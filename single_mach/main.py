"""Single Mach example.

This experiment demonstrates a single linear model and a student who learns to
approximate its embedding, we test the resulting network by having the student
output its error on the test set using the student model.

Example:
        Train the model over 100000 iterations.
        $ python single_mach.py --n_iterations=100000

Todo:
    * CIFAR
"""

import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def load_data_and_constants(hparams):
    '''Returns the dataset and sets hparams.n_inputs and hparamsn_targets.'''
    # Load mnist data
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    hparams.n_inputs = 784
    hparams.n_targets = 10
    return mnist, hparams


def inputs(hparams):
    '''Builds tensorflow input and targer placeholders.'''
    x_inputs = tf.placeholder("float", [None, hparams.n_inputs], 'inputs')
    y_targets = tf.placeholder("float", [None, hparams.n_targets], 'targets')
    return x_inputs, y_targets


def teacher(x_inputs, hparams):
    '''Builds the teacher model, returns the teacher's embedding'''
    weights = {
        'w1':
            tf.Variable(
                tf.truncated_normal([hparams.n_inputs, hparams.t_hidden1],
                                    stddev=0.1)),
        'w2':
            tf.Variable(
                tf.truncated_normal([hparams.t_hidden1, hparams.t_hidden2],
                                    stddev=0.1)),
        'w3':
            tf.Variable(
                tf.truncated_normal([hparams.t_hidden2, hparams.n_embedding],
                                    stddev=0.1)),
    }

    biases = {
        'b1': tf.Variable(tf.constant(0.1, shape=[hparams.t_hidden1])),
        'b2': tf.Variable(tf.constant(0.1, shape=[hparams.t_hidden2])),
        'b3': tf.Variable(tf.constant(0.1, shape=[hparams.n_embedding])),
    }

    layer_1 = tf.nn.relu(
        tf.add(tf.matmul(x_inputs, weights['w1']), biases['b1']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['w2']),
                                biases['b2']))
    teacher_embedding = tf.nn.relu(
        tf.add(tf.matmul(layer_2, weights['w3']), biases['b3']))

    return teacher_embedding


def student(x_inputs, hparams):
    '''Builds the student model, returns the students's embedding'''
    weights = {
        'w1':
            tf.Variable(
                tf.truncated_normal([hparams.n_inputs, hparams.s_hidden1],
                                    stddev=0.1)),
        'w2':
            tf.Variable(
                tf.truncated_normal([hparams.s_hidden1, hparams.s_hidden2],
                                    stddev=0.1)),
        'w3':
            tf.Variable(
                tf.truncated_normal([hparams.s_hidden2, hparams.n_embedding],
                                    stddev=0.1)),
    }

    biases = {
        'b1': tf.Variable(tf.constant(0.1, shape=[hparams.s_hidden1])),
        'b2': tf.Variable(tf.constant(0.1, shape=[hparams.s_hidden2])),
        'b3': tf.Variable(tf.constant(0.1, shape=[hparams.n_embedding])),
    }

    layer_1 = tf.nn.relu(
        tf.add(tf.matmul(x_inputs, weights['w1']), biases['b1']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['w2']),
                                biases['b2']))
    student_embedding = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])

    return student_embedding


def logits(t_embedding, s_embedding, hparams):
    '''Calculates the teacher and student logits from embeddings.'''
    w = tf.Variable(
        tf.truncated_normal([hparams.n_embedding, hparams.n_targets],
                            stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[hparams.n_targets])),
    t_logits = tf.add(tf.matmul(t_embedding, w), b)
    s_logits = tf.add(tf.matmul(s_embedding, w), b)
    return t_logits, s_logits


def target_loss(logits, targets, hparams):
    '''Calculates the target loss w.r.t a set of logits.'''
    target_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets,
                                                   logits=logits))
    return target_loss


def distillation_loss(student_embedding, teacher_embedding, hparams):
    '''Calculates the distilled loss between the teacher and student embedding'''
    distillation_loss = tf.reduce_mean(
        tf.nn.l2_loss(tf.stop_gradient(teacher_embedding) - student_embedding))
    return distillation_loss


def make_metrics(teacher_embedding, student_embedding, teacher_logits,
                 student_logits, teacher_loss, student_loss, dist_loss,
                 y_targets, x_inputs, hparams):
    '''Builds and returns all model metrics as a dictionary '''

    t_correct = tf.equal(tf.argmax(teacher_logits, 1), tf.argmax(y_targets, 1))
    t_accuracy = tf.reduce_mean(tf.cast(t_correct, tf.float32))

    s_correct = tf.equal(tf.argmax(student_logits, 1), tf.argmax(y_targets, 1))
    s_accuracy = tf.reduce_mean(tf.cast(s_correct, tf.float32))

    metrics = {
        'tloss': teacher_loss,
        'sloss': student_loss,
        'tacc': t_accuracy,
        'sacc': s_accuracy,
        'distloss': dist_loss
    }

    return metrics


def training_step(metrics, hparams):
    '''Returns training step. '''
    full_loss = metrics['tloss'] + metrics['distloss']
    return tf.train.AdamOptimizer(hparams.learning_rate).minimize(full_loss)


def train(session, mnist, step, metrics, hparams):
    ''' Trains model for hprams.n_iterations printing metrics every hparams.n_print '''

    # Training loop.
    for i in range(hparams.n_iterations):
        batch_x, batch_y = mnist.train.next_batch(hparams.batch_size)
        feeds = {'inputs:0': batch_x, 'targets:0': batch_y}
        session.run(step, feeds)

        # Print metrics.
        if i % hparams.n_print == 0:
            feeds = {'inputs:0': batch_x, 'targets:0': batch_y}
            train_metrics = session.run(metrics, feeds)
            print(train_metrics)


def test(session, mnist, metrics, hparams):
    ''' Tests model. Printing student and teacher accuracy. '''
    feeds = {'inputs:0': mnist.test.images, 'targets:0': mnist.test.labels}
    test_metrics = session.run(metrics, feeds)
    print("\nTeacher accuracy on test set:", test_metrics['tacc'])
    print("\nStudent accuracy on test set:", test_metrics['sacc'])


def main(hparams):

    # Best with Defaults:
    # Teacher Accuracy 0.9787
    # Student Accuracy: 0.9749

    # Load dataset and set hparams.n_inputs and hparams.n_targets.
    mnist, hparams = load_data_and_constants(hparams)

    # Build graph and session.
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default():

        # Build tensorflow input and target placeholders.
        x_inputs, y_targets = inputs(hparams)

        # Build the teacher model, returns the teacher's embedding
        teacher_embedding = teacher(x_inputs, hparams)

        # Build the student model, returns the students's embedding
        student_embedding = student(x_inputs, hparams)

        # Calculate the teacher and student logits from embeddings.
        teacher_logits, student_logits = logits(teacher_embedding,
                                                student_embedding, hparams)

        # Calculate the target loss w.r.t teacher logits.
        teacher_loss = target_loss(teacher_logits, y_targets, hparams)

        # Calculate the target loss w.r.t student logits.
        student_loss = target_loss(tf.stop_gradient(student_logits), y_targets,
                                   hparams)

        # Calculate the distilled loss between the teacher and student embedding
        dist_loss = distillation_loss(student_embedding, teacher_embedding,
                                      hparams)

        # Build and return all model metrics as a dictionary
        metrics = make_metrics(teacher_embedding, student_embedding,
                               teacher_logits, student_logits, teacher_loss,
                               student_loss, dist_loss, y_targets, x_inputs,
                               hparams)

        # Build training step from metrics.
        train_step = training_step(metrics, hparams)

        # Init graph.
        session.run(tf.global_variables_initializer())

    # Train model for hprams.n_iterations printing metrics every hparams.n_print
    train(session, mnist, train_step, metrics, hparams)

    # Test model. Print student and teacher accuracy.
    test(session, mnist, metrics, hparams)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch_size',
        default=128,
        type=int,
        help='The number of examples per batch. Default batch_size=128')
    parser.add_argument(
        '--n_iterations',
        default=10000,
        type=int,
        help='The number of training iterations. Default n_iterations=10000')
    parser.add_argument(
        '--n_print',
        default=100,
        type=int,
        help='The number of iterations before metrics print. Default n_print=100'
    )
    parser.add_argument(
        '--learning_rate',
        default=1e-4,
        type=float,
        help='Network learning rate. Default learning_rate=1e-4')
    parser.add_argument(
        '--n_embedding',
        default=128,
        type=int,
        help=
        'Size of teacher and student embedding layer. Default n_embedding=128')
    parser.add_argument(
        '--t_hidden1',
        default=512,
        type=int,
        help='Size of teacher first layer. Default t_hidden1=512')
    parser.add_argument(
        '--t_hidden2',
        default=256,
        type=int,
        help='Size of teacher second layer. Default t_hidden2=256')
    parser.add_argument(
        '--s_hidden1',
        default=512,
        type=int,
        help='Size of student first layer. Default s_hidden1=512')
    parser.add_argument(
        '--s_hidden2',
        default=256,
        type=int,
        help='Size of student second layer. Default s_hidden2=256')

    hparams = parser.parse_args()

    main(hparams)
