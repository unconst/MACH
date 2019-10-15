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
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    hparams.n_inputs = 784
    hparams.n_targets = 10
    return mnist, hparams


def inputs(hparams):
    '''Builds tensorflow input and targer placeholders.'''
    x_inputs = tf.placeholder("float", [None, hparams.n_inputs], 'inputs')
    y_targets = tf.placeholder("float", [None, hparams.n_targets], 'targets')
    return x_inputs, y_targets


def teacher(i, x_inputs, student_embedding, hparams):
    '''Builds the teacher model, returns the teacher's embedding'''

    teacher_inputs = tf.concat([x_inputs, student_embedding], axis=1)
    n_teacher_inputs = hparams.n_inputs + hparams.n_embedding

    weights = {
        'w1':
            tf.Variable(
                tf.truncated_normal([n_teacher_inputs, hparams.t_hidden1],
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
        tf.add(tf.matmul(teacher_inputs, weights['w1']), biases['b1']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['w2']),
                                biases['b2']))
    teacher_embedding = tf.nn.relu(
        tf.add(tf.matmul(layer_2, weights['w3']), biases['b3']))

    if hparams.trace:
        teacher_embedding = tf.Print(teacher_embedding, [teacher_embedding],
                                     summarize=10,
                                     message='teacher_embedding_' + str(i))

    return teacher_embedding


def student(i, x_inputs, hparams):
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

    if hparams.trace:
        student_embedding = tf.Print(student_embedding, [student_embedding],
                                     summarize=10,
                                     message='student_embedding_' + str(i))

    return student_embedding


def logits(embedding, hparams):
    '''Calculates the teacher and student logits from embeddings.'''
    w = tf.Variable(
        tf.truncated_normal([hparams.n_embedding, hparams.n_targets],
                            stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[hparams.n_targets])),
    logits = tf.add(tf.matmul(embedding, w), b)
    return logits


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


def add_metrics(i, metrics, student_embedding, teacher_embedding,
                teacher_logits, teacher_loss, student_loss, y_targets, x_inputs,
                hparams):
    '''Builds and returns all model metrics as a dictionary '''

    t_correct = tf.equal(tf.argmax(teacher_logits, 1), tf.argmax(y_targets, 1))
    t_accuracy = tf.reduce_mean(tf.cast(t_correct, tf.float32))

    metrics['tloss_' + str(i)] = teacher_loss
    metrics['tacc_' + str(i)] = t_accuracy
    metrics['sloss_' + str(i)] = student_loss

    return metrics


def training_step(metrics, hparams):
    '''Returns training step. '''
    # Aggregate losses from each component.
    for i in range(hparams.n_components):
        tf.compat.v1.losses.add_loss(metrics['tloss_' + str(i)])
        tf.compat.v1.losses.add_loss(metrics['sloss_' + str(i)])
    loss = tf.compat.v1.losses.get_total_loss()
    return tf.compat.v1.train.AdamOptimizer(
        hparams.learning_rate).minimize(loss)


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
    print("\nResults:")
    for i in range(hparams.n_components):
        print("Teacher_" + str(i) + " accuracy on test set:",
              test_metrics['tacc_' + str(i)])


def main(hparams):

    # Testing framework for MACHs.
    # We build our graph as a sequence of components. Each component contains
    # a teacher model and a student. However, the student is training off the
    # previous teacher rather than the teacher in the current component.
    # The teacher uses this distilled student model to pull information from the
    # previous teacher. Since the distilled model is 'local' we don't need to
    # run the entire preceding network during inference.

    # Best with Defaults Size = 2; iterations=10000
    # Teacher_0 accuracy on test set: 0.9802
    # Teacher_1 accuracy on test set: 0.9803

    # Best with Defaults Size = 3; iterations=10000
    # Teacher_0 accuracy on test set: 0.9775
    # Teacher_1 accuracy on test set: 0.979
    # Teacher_2 accuracy on test set: 0.979

    # Load dataset and set hparams.n_inputs and hparamsn_targets.
    mnist, hparams = load_data_and_constants(hparams)

    # Build graph and session.
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default():

        # Build tensorflow input and target placeholders.
        x_inputs, y_targets = inputs(hparams)

        # Placeholder for the previous component embedding. Set to zeros for the
        # first component. The prev_teacher will point to the embedding layer
        # from the previous model in following components.
        prev_teacher = tf.zeros([tf.shape(x_inputs)[0], hparams.n_embedding])

        # Stored metrics from each component.
        metrics = {}

        # Build component hierarchy.
        for i in range(hparams.n_components):

            # Build the student model, returns the students's embedding
            # The student model is a FF-NN which trains itself to predict the
            # embedding from previous model.
            student_embedding = student(i, x_inputs, hparams)

            # Calculate the distilled loss between the prev_embedding and the student's
            student_loss = distillation_loss(student_embedding, prev_teacher,
                                             hparams)

            # Build the teacher model, returns the teacher's embedding
            # Note to stop gradient between the teacher and the student.
            teacher_embedding = teacher(i, x_inputs,
                                        tf.stop_gradient(student_embedding),
                                        hparams)

            # Calculate the teacher logits from embeddings.
            teacher_logits = logits(teacher_embedding, hparams)

            # Calculate the target loss w.r.t teacher logits.
            teacher_loss = target_loss(teacher_logits, y_targets, hparams)

            # Build and return all model metrics as a dictionary
            metrics = add_metrics(i, metrics, student_embedding,
                                  teacher_embedding, teacher_logits,
                                  teacher_loss, student_loss, y_targets,
                                  x_inputs, hparams)

            # The teacher's output becomes the next prev_embedding.
            prev_teacher = teacher_embedding

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
        '--n_components',
        default=1,
        type=int,
        help='Number of components to train. Default n_components=1')
    parser.add_argument(
        '--trace',
        default=False,
        type=bool,
        help='Print embeddings from each layer. Default trace=False')
    parser.add_argument(
        '--stop_grad_for_student',
        default=False,
        type=bool,
        help=
        'Stop training grad for student learning on logits. Default trace=False'
    )
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
