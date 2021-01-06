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


def blackbox_equals(m1, m2, produce_input, device: torch.device = None):
    """
    Compares two models in a blackbox manner meaning if the models are equal is determined only by comparing inputs and
    outputs.
    :param m1: The first model to compare.
    :param m2: The second model to compare.
    :param produce_input: Method to produce input accepted by the given models.
    :param device: The device to execute on
    :return: Returns if the two given models are equal.
    """

    inp = produce_input()

    if 'cuda' in str(device):
        m1.to(device)
        m2.to(device)
        inp = inp.cuda()

    m1.eval()
    m2.eval()

    out1 = m1(inp)
    out2 = m2(inp)

    return torch.equal(out1, out2)


def whitebox_equals(m1, m2, device: torch.device = None):
    """
    Compares two models in a whitebox manner meaning we compare the model weights.
    :param m1: The first model to compare.
    :param m2: The second model to compare.
    :param device: The device to execute on
    :return: Returns if the two given models are equal.
    """
    state1 = m1.state_dict()
    state2 = m2.state_dict()

    return state_dict_equals(state1, state2, device)


def state_dict_equals(d1, d2, device: torch.device = None):
    """
    Compares two given state dicts.
    :param d1: The first state dict.
    :param d2: The first state dict.
    :param device: The device to execute on
    :return: Returns if the given state dicts are equal.
    """

    for item1, item2 in zip(d1.items(), d2.items()):
        layer_name1, weight_tensor1 = item1
        layer_name2, weight_tensor2 = item2

        # TODO USE GENERIC DEVICE
        if 'cuda' in str(device):
            weight_tensor1 = weight_tensor1.cuda()
            weight_tensor2 = weight_tensor2.cuda()

        if not layer_name1 == layer_name2 or not torch.equal(weight_tensor1, weight_tensor2):
            return False

    return True


def equals(m1, m2, produce_input, device: torch.device = None):
    """
    An equals method to compare two given models by making use of whitebox and blackbox equals.
    :param m1: The first model to compare.
    :param m2: The second model to compare.
    :param produce_input: Method to produce input accepted by models processing imagenet data.
    :param device: The device to execute on
    :return: Returns if the two given models are equal.
    """
    # whitebox and blackbox check should be redundant,
    # but this way we have an extra safety net in case we forgot a special case
    return whitebox_equals(m1, m2, device) and blackbox_equals(m1, m2, produce_input, device)
