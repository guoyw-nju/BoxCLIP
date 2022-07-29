import os
import yaml
import argparse

from traitlets import default

def add_training_options(parser):
    group = parser.add_argument_group('Training options')

    group.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    group.add_argument("--num_epochs", type=int, default=20, help="number of epochs of training")
    group.add_argument("--lr", type=float, default=0.0001, help="AdamW: learning rate")
    group.add_argument("--lr_gamma", type=float, default=0.5, help="for Lr step scheduler")
    group.add_argument("--overfit", default=False, action="store_true", help="enable overfit training test")
    group.add_argument("--overfit_size", type=int, default=20, help="size of the overfit training set")

    group.add_argument("--folder", type=str, default="./checkpoint", help="folder to save the checkpoint")
    group.add_argument("--exp_name", type=str, help="name of the experiment")

    group.add_argument('--checkpoint_path', type=str, help ='whether train on an existing model')


def parser():
    parser = argparse.ArgumentParser()

    add_training_options(parser)

    args = parser.parse_args()

    parameters = {k: v for k, v in vars(args).items()}

    parameters['checkpoint_path'] = os.join(parameters['folder'], parameters['exp_name'])

    os.makedirs(parameters['checkpoint_path'], exist_ok=True)
    optpath = os.path.join(parameters['checkpoint_path'], "opt.yaml")
    with open(optpath, 'w') as opt_file:
        yaml.dump(parameters, opt_file)

    return parameters

