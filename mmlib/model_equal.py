import torch

from util.helper import _get_device


def blackbox_equal(m1, m2, produce_input, device: torch.device = None):
    """
    Compares two models in a blackbox manner meaning if the models are equal is determined only by comparing inputs and
    outputs.
    :param m1: The first model to compare.
    :param m2: The second model to compare.
    :param produce_input: Method to produce input accepted by the given models.
    :param device: The device to execute on.
    :return: Returns if the two given models are equal.
    """

    assert isinstance(m1, torch.nn.Module)
    assert isinstance(m2, torch.nn.Module)

    device = _get_device(device)

    inp = produce_input()

    m1.to(device)
    m2.to(device)
    inp = inp.to(device)

    m1.eval()
    m2.eval()

    out1 = m1(inp)
    out2 = m2(inp)

    return torch.equal(out1, out2)


def whitebox_equal(m1, m2, device: torch.device = None):
    """
    Compares two models in a whitebox manner meaning we compare the model weights.
    :param m1: The first model to compare.
    :param m2: The second model to compare.
    :param device: The device to execute on.
    :return: Returns if the two given models are equal.
    """

    assert isinstance(m1, torch.nn.Module)
    assert isinstance(m2, torch.nn.Module)

    device = _get_device(device)

    state1 = m1.state_dict()
    state2 = m2.state_dict()

    return state_dict_equal(state1, state2, device)


def state_dict_equal(d1, d2, device: torch.device = None):
    """
    Compares two given state dicts.
    :param d1: The first state dict.
    :param d2: The first state dict.
    :param device: The device to execute on
    :return: Returns if the given state dicts are equal.
    """

    device = _get_device(device)

    for item1, item2 in zip(d1.items(), d2.items()):
        layer_name1, weight_tensor1 = item1
        layer_name2, weight_tensor2 = item2

        weight_tensor1 = weight_tensor1.to(device)
        weight_tensor2 = weight_tensor2.to(device)

        if not layer_name1 == layer_name2 or not torch.equal(weight_tensor1, weight_tensor2):
            return False

    return True


def equal(m1, m2, produce_input, device: torch.device = None):
    """
    An equals method to compare two given models by making use of whitebox and blackbox equals.
    :param m1: The first model to compare.
    :param m2: The second model to compare.
    :param produce_input: Method to produce input accepted by models processing imagenet data.
    :param device: The device to execute on
    :return: Returns if the two given models are equal.
    """
    device = _get_device(device)

    # whitebox and blackbox check should be redundant,
    # but this way we have an extra safety net in case we forgot a special case
    return whitebox_equal(m1, m2, device) and blackbox_equal(m1, m2, produce_input, device)
