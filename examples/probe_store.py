import argparse
import os

import torch
from torch import nn
from torchvision import models

from mmlib.deterministic import set_deterministic
from mmlib.probe import probe_training
from mmlib.util.dummy_data import imagenet_input, imagenet_target


def main(args):
    # we create a probe summary and get it back as an object
    summary = _generate_probe_training_summary()
    # we can save the summary to the path given in the args
    output_path = os.path.join(args.path, 'summary')
    summary.save(output_path)


def _generate_probe_training_summary():
    # First, we force the implementation to be deterministic using mmlib's set_deterministic() function
    set_deterministic()
    # as an example we want to prob the GoogLeNet architecture
    model = models.googlenet(pretrained=True)
    # to probe tha forward and backward pass we have to create some dummy data
    # we need: input, target, loss function and optimizer
    dummy_input = imagenet_input()
    dummy_target = imagenet_target(dummy_input)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 1e-3)
    # having created the model and all dummy data we can execute a probe run and return the summary
    summary = probe_training(model, dummy_input, optimizer, loss_func, dummy_target)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description='Creat and store a probe Summary')
    parser.add_argument('--path', help='path to store summary data to', default='.')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
