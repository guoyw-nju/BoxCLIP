import argparse

from traitlets import default

def add_training_options(parser):
    group = parser.add_argument_group('Training options')

    group.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    group.add_argument("--num_epochs", type=int, default=20, help="number of epochs of training")
    group.add_argument("--lr", type=float, default=0.0001, help="AdamW: learning rate")
    parser.add_argument('--checkpoint_path', type=str, help ='whether train on an existing model')


def parser():
    parser = argparse.ArgumentParser()

    add_training_options(parser)

    args = parser.parse_args()

    return {k: v for k, v in vars(args).items()}

