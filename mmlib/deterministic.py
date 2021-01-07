import os
import random

import numpy as np
import torch

SEED = 42


def deterministic(func, f_args=None, f_kwargs=None):
    """
    Executed the given function in a deterministic calling set_deterministic before
    :param func: The function to execute.
    :param f_args: The args for the function to execute.
    :param f_kwargs: The kwargs for the function to execute.
    :return: The results of the executed function.
    """
    if f_kwargs is None:
        f_kwargs = {}
    if f_args is None:
        f_args = []

    set_deterministic()
    return func(*f_args, **f_kwargs)


def set_deterministic():
    """
    Makes execution reproducible following the instructions form:
    https://pytorch.org/docs/1.7.1/notes/randomness.html?highlight=reproducibility
    """
    # set seeds for pytorch and all used libraries
    random.seed(SEED)  # seed for random python
    torch.manual_seed(SEED)  # seed the RNG for all devices (both CPU and CUDA)
    np.random.seed(SEED)

    # turn of benchmarking for convolutions
    torch.backends.cudnn.benchmark = False

    # avoid non-deterministic algorithms
    torch.set_deterministic(True)

    # for CUDA version10.2 or greater: set the environment variable CUBLAS_WORKSPACE_CONFIG according to CUDA
    # documentation: https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    # set a debug environment variable CUBLAS_WORKSPACE_CONFIG to
    # ":16:8" (may limit overall performance) or
    # ":4096:8" (will increase library footprint in GPU memory by approximately 24MiB).
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"
