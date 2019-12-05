"""Run the trainig pipelie.

Example:
        $ python main.py --batch_size 100

"""

import argparse
from loguru import logger
import threading
import time

from moe_model import Mach
from utils import TBLogger
from utils import metagraph_summary
from utils import next_run_prefix
from utils import load_data_and_constants


def run(hparams, run_prefix, tblogger, components):
    for c in components:
        c.start()
    logger.info('Begin wait on main...')
    running = True

    while running:
        for c in components:
            if c.running == False:
                running = False
        metagraph_summary(components, tblogger, run_prefix, components[0].step, hparams)

    for c in components:
        c.stop()

def main(hparams):

    # Get a unique id for this training run.
    run_prefix = next_run_prefix()

    # Build components.
    components = []
    for i in range(hparams.n_components):
        # Load a unique dataset for each component.
        mnist_i, hparams = load_data_and_constants(hparams)

        # Tensorboard logger tool.
        logdir_i = hparams.log_dir + "/" + run_prefix + "/" + 'c' + str(i)
        tblogger_i = TBLogger(logdir_i)

        # Component.
        mach_i = Mach(i, mnist_i, hparams, tblogger_i)
        components.append(mach_i)

    # Connect components
    for i in range(hparams.n_components):
        components[i].set_children(components[:i] + components[i+1:])

    # metagraph logger
    logdir = hparams.log_dir + "/" + run_prefix + "/" + 'm'
    tblogger = TBLogger(logdir)

    # Run experiment.
    run(hparams, run_prefix, tblogger, components)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch_size',
        default=50,
        type=int,
        help='The number of examples per batch. Default batch_size=128')
    parser.add_argument(
        '--learning_rate',
        default=1e-5,
        type=float,
        help='Component learning rate. Default learning_rate=1e-4')
    parser.add_argument(
        '--n_embedding',
        default=128,
        type=int,
        help='Size of embedding between components. Default n_embedding=128')
    parser.add_argument(
        '--n_components',
        default=3,
        type=int,
        help='The number of training iterations. Default n_components=2')
    parser.add_argument(
        '--k',
        default=2,
        type=int,
        help='Number to components to query. Default k=2')
    parser.add_argument(
        '--n_iterations',
        default=10000,
        type=int,
        help='The number of training iterations. Default n_iterations=10000')
    parser.add_argument(
        '--n_hidden1',
        default=512,
        type=int,
        help='Size of layer 1. Default n_hidden1=512')
    parser.add_argument(
        '--n_hidden2',
        default=512,
        type=int,
        help='Size of layer 1. Default n_hidden2=512')
    parser.add_argument(
        '--n_shidden1',
        default=512,
        type=int,
        help='Size of synthetic model hidden layer 1. Default n_shidden1=512')
    parser.add_argument(
        '--n_shidden2',
        default=512,
        type=int,
        help='Size of synthetic model hidden layer 2. Default n_shidden2=512')
    parser.add_argument(
        '--max_depth',
        default=1,
        type=int,
        help='Depth at which the synthetic inputs are used. Default max_depth=2')
    parser.add_argument(
        '--n_print',
        default=100,
        type=int,
        help=
        'The number of iterations between print statements. Default n_print=100'
    )
    parser.add_argument(
        '--log_dir',
        default='logs',
        type=str,
        help='location of tensorboard logs. Default log_dir=logs'
    )
    parser.add_argument(
        '--n_train_steps',
        default=10000000,
        type=int,
        help='Training steps. Default n_train_steps=1000000'
    )

    hparams = parser.parse_args()

    main(hparams)
