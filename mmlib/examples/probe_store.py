import argparse
import os

import torch
from torch import nn
from torchvision import models

from mmlib.deterministic import set_deterministic
from mmlib.probe import probe_training
from mmlib.util.dummy_data import imagenet_input, imagenet_target


def main(args):
    summary = _generate_probe_training_summary()

    output_path = os.path.join(args.path, 'summary')
    summary.save(output_path)


def _generate_probe_training_summary():
    set_deterministic()
    dummy_input = imagenet_input()
    dummy_target = imagenet_target(dummy_input)
    loss_func = nn.CrossEntropyLoss()
    model = models.googlenet(pretrained=True)
    optimizer = torch.optim.SGD(model.parameters(), 1e-3)
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
