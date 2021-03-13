import hashlib

import torch

from mmlib.deterministic import set_deterministic
from util.helper import get_device


def inference_hash(model: torch.nn.Module, dummy_input_shape: [int]):
    """
    Calculates the hash of an inference produced by the given model.
    :param model: The model that is used for inference.
    :param dummy_input_shape: The input shape for the inference.
    :return: The hash of the produced inference.
    """
    set_deterministic()
    model.eval()
    dummy_input = torch.rand(dummy_input_shape)
    dummy_output = model(dummy_input)
    inference_hash = tensor_hash(dummy_output)
    return inference_hash


def state_dict_hash(state_dict: dict, device: torch.device = None) -> str:
    """
    Calculates a md5 hash of a state dict dependent on the layer names and the corresponding weight tensors.
    :param state_dict: The state dict to create the hash from.
    :param device: The device to execute on.
    :return: The md5 hash as a string.
    """
    md5 = hashlib.md5()

    device = get_device(device)

    for layer_name, weight_tensor in state_dict.items():
        weight_tensor = weight_tensor.to(device)
        numpy_data = weight_tensor.numpy().data
        md5.update(bytes(layer_name, 'utf-8'))
        md5.update(numpy_data)

    return md5.hexdigest()


def tensor_hash(tensor: torch.tensor, device: torch.device = None) -> str:
    """
    Calculates a md5 hash of the given tensor.
    :param tensor: The tensor to hash.
    :param device: The device to execute on.
    :return: The md5 hash as a string.
    """
    md5 = hashlib.md5()

    device = get_device(device)

    tensor = tensor.to(device)
    numpy_data = tensor.detach().numpy().data
    md5.update(numpy_data)

    return md5.hexdigest()
