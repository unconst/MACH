from loguru import logger
import numpy as np
import random
import io
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from fa2 import ForceAtlas2

def load_data_and_constants(hparams):
    '''Returns the dataset and sets hparams.n_inputs and hparamsn_targets.'''
    # Load mnist data
    mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True)
    hparams.n_inputs = 784
    hparams.n_targets = 10
    return mnist, hparams

def next_nounce():
    return random.randint(0, 1000000000)

def next_run_prefix():
    return str(int(time.time()))

def _networkx(components):
    G = nx.DiGraph()

    node_labels = {}
    node_sizes = []
    for c in components:
        G.add_node(c.name)
        node_labels[c.name] = str(c.name)
        node_sizes.append(0.1 + c.revenue)

    edge_labels = {}
    for parent in components:
        for child in components:
            G.add_edge(parent.name, child.name)
            edge_labels[(parent.name, child.name)] = "%.3f" % parent.weights[child.name]

    forceatlas2 = ForceAtlas2(
                            # Behavior alternatives
                            outboundAttractionDistribution=True,  # Dissuade hubs
                            linLogMode=False,  # NOT IMPLEMENTED
                            adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                            edgeWeightInfluence=1.0,

                            # Performance
                            jitterTolerance=1.0,  # Tolerance
                            barnesHutOptimize=True,
                            barnesHutTheta=1.2,
                            multiThreaded=False,  # NOT IMPLEMENTED

                            # Tuning
                            scalingRatio=2.0,
                            strongGravityMode=False,
                            gravity=1.0,

                            # Log
                            verbose=False)

    positions = nx.layout.circular_layout(G)
    pos_higher = {}
    y_off = 0.2
    for k, v in positions.items():
        pos_higher[k] = (v[0], v[1]+y_off)

    nx.draw_networkx_nodes(G, positions, with_labels=True, node_size=node_sizes, node_color="blue", alpha=0.4)
    nx.draw_networkx_edges(G, positions, arrowstyle='->', arrowsize=15, edge_color="green", edge_labels=edge_labels, alpha=0.05, label_pos=0.3)
    nx.draw_networkx_labels(G, pos_higher, node_labels)
    nx.draw_networkx_edge_labels(G, pos_higher, edge_labels=edge_labels, with_labels=True, label_pos=0.3)

def metagraph_summary(components, tblogger, run_prefix, step, hparams):
    figure = plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.savefig(hparams.log_dir + "/" + run_prefix + str('/metagraph'))
    _networkx(components)
    tblogger.log_plot('metagraph', step)
    plt.close()

class TBLogger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        summary = tf.Summary(
            value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def log_plot(self, tag, step):

        output = io.BytesIO()
        plt.savefig(output, format='png')
        image_string = output.getvalue()
        output.close()

        # Create an Image object
        img_sum = tf.Summary.Image(encoded_image_string=image_string,
                                   height=10,
                                   width=10)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, image=img_sum)])
        self.writer.add_summary(summary, step)


    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
