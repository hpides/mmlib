import random

import torch


def imagenet_input(batch_size=10):
    """
    Generates a batch of dummy imputes for models processing imagenet data.
    :param batch_size: The size of the batch.
    :return: Returns a tensor containing the generated batch.
    """
    batch = []
    for i in range(batch_size):
        batch.append(torch.rand(3, 300, 400))
    return torch.stack(batch)


def imagenet_target(dummy_input):
    """
    Creates a batch of random labels for imagenet data based on a given input data.
    :param dummy_input: The input to a potential model for the the target values should be produced.
    :return: The batch of random targets.
    """
    batch_size = dummy_input.shape[0]
    batch = []
    for i in range(batch_size):
        batch.append(random.randint(1, 999))
    return torch.tensor(batch)
