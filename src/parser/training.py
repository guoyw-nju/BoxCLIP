import os
import yaml
import argparse
import torch


def add_training_options(parser):
    group = parser.add_argument_group('Training options')

    # training setttings
    group.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    group.add_argument("--num_epochs", type=int, default=20, help="number of epochs of training")

    group.add_argument("--lr", type=float, default=0.0001, help="AdamW: learning rate")
    group.add_argument("--lr_scheduler", type=str, default=None, choices=['ReduceLROnPlateau'], help="type of the lr scheduler")

    group.add_argument("--overfit", default=False, action="store_true", help="enable overfit training test")
    group.add_argument("--overfit_size", type=int, default=20, help="size of the overfit training set")
    group.add_argument("--no_val_loss", default=False, action="store_true", help="do not draw loss curve on val set")
    group.add_argument("--checkpoint_step", type=int, default=5, help="save checkpoint per [step] epochs")

    # model
    group.add_argument("--num_attentionLayer", type=int, default=1, help="num of attention layer of the encoder/decoder")
    # losses weight
    group.add_argument("--clip_image_cosine", type=float, default=0, help="weight for clip image cosine loss")
    group.add_argument("--clip_text_cosine", type=float, default=0, help="weight for clip text cosine loss")
    group.add_argument("--bbox_mse", type=float, default=1, help="weight for bounding box corrdinate reconstruction mse loss")
    group.add_argument("--cats_cos", type=float, default=1, help="weight for bounding box category reconstruction cosine loss")
    group.add_argument("--bbox_mse_gen", type=float, default=0, help="weight for bounding box corrdinate reconstruction mse loss")
    group.add_argument("--cats_cos_gen", type=float, default=0, help="weight for bounding box category reconstruction cosine loss")

    # experiment settings
    group.add_argument("--folder", type=str, default="./checkpoint", help="folder to save the checkpoint")
    group.add_argument("--exp_name", type=str, help="name of the experiment")
    group.add_argument('--checkpoint_path', type=str, help ='whether train on an existing model')


def parser():
    parser = argparse.ArgumentParser()

    add_training_options(parser)

    args = parser.parse_args()

    parameters = {k: v for k, v in vars(args).items()}
    if torch.cuda.is_available():
        parameters.update({'device': 'cuda'})
    else:
        parameters.update({'device': 'cpu'})

    # losses weight
    lambdas = {}
    lambdas['clip_image_cosine'] = parameters['clip_image_cosine']
    lambdas['clip_text_cosine'] = parameters['clip_text_cosine']
    lambdas['bbox_mse'] = parameters['bbox_mse']
    lambdas['cats_cos'] = parameters['cats_cos']
    lambdas['bbox_mse_gen'] = parameters['bbox_mse_gen']
    lambdas['cats_cos_gen'] = parameters['cats_cos_gen']
    parameters['lambdas'] = lambdas
    

    parameters['exp_path'] = os.path.join(parameters['folder'], parameters['exp_name'])

    print('Training Parameters:\n', parameters)
    os.makedirs(parameters['exp_path'], exist_ok=True)
    optpath = os.path.join(parameters['exp_path'], "opt.yaml")
    with open(optpath, 'w') as opt_file:
        yaml.dump(parameters, opt_file)

    return parameters

